import torch
import torch.nn as nn
import numpy as np

#%%%%%%% Helper Functions %%%%%%%%#

def pairwise(x):
    ''' input x = (eta, cos(phi), sin(phi)), output matrix xij = (eta_i - eta_j, cos(phi_i - phi_j), sin(phi_i - phi_j)) '''
    eta, cphi, sphi = x.split((1, 1, 1), dim=-1)
    eta_ij = cast(eta, eta, "subtract")
    sin_ij = cast(sphi, cphi, "times") - cast(cphi, sphi, "times")
    cos_ij = cast(cphi, cphi, "times") + cast(sphi, sphi, "times")
    xij = torch.stack([eta_ij, cos_ij, sin_ij], -1)
    return xij

def cast(a, b, op="add"):
    ''' a and b are batched (NEvents, NJets, NObs) '''
    a = a.repeat(1, 1, a.shape[1])
    b = b.repeat(1, 1, b.shape[1]).transpose(2,1)
    if op == "add":
        return a + b
    elif op == "subtract":
        return a - b
    elif op == "times":
        return a*b
    return a+b

#%%%%%%% Classes %%%%%%%%#

class DNN_block(nn.Module):

    def __init__(self, dimensions, normalize_input):

        super().__init__()
        input_dim = dimensions[0]
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None

        layers = []
        for iD, (dim_in, dim_out) in enumerate(zip(dimensions, dimensions[1:])):
            if iD != len(dimensions[1:])-1:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                    nn.LayerNorm(dim_out),
                    nn.ReLU(),
                ])
            else:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_bn is not None:
            x = self.input_bn(x.transpose(1,2)) # must transpose because batchnorm expects N,C,L rather than N,L,C like multihead
            x = x.transpose(1,2) # revert transpose
        return self.net(x)

class AttnBlock(nn.Module):

    def __init__(self,
                 embed_dim = 4,
                 num_heads = 1,
                 attn_dropout = 0.1,
                 add_bias_kv = True,
                 kdim = None,
                 vdim = None,
                 ffwd_dims = [16,16],
                 ffwd_dropout = 0):
        super().__init__()
        
        # multihead attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv, batch_first=True, kdim=kdim, vdim=vdim)

        # normalization after attn
        self.post_attn_norm = nn.LayerNorm(embed_dim)

        # feed forward
        self.ffwd, input_dim = [], embed_dim
        for i, dim in enumerate(ffwd_dims):
            self.ffwd.extend([
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            input_dim = dim
        self.ffwd = nn.Sequential(*self.ffwd)
        
        # normalization after ffwd
        self.post_ffwd_norm = nn.LayerNorm(ffwd_dims[-1]) if ffwd_dims[-1] == embed_dim else None        

    def forward(self, Q, K, V, key_padding_mask, attn_mask):
        ''' 
        Input is (Batch, Jet, Embedding Dim) = (B,J,E) 
        Output is (B,J,E)
        '''

        residual = V
        
        # attention
        V, _ = self.attn(query=Q, key=K, value=V, key_padding_mask=key_padding_mask, attn_mask = attn_mask, need_weights=False)

        # skip connection & norm
        if V.shape == residual.shape:
            V = V + residual
        V = self.post_attn_norm(V)

        # feed forward & skip connection & norm
        residual = V
        V = self.ffwd(V)
        V = V + residual
        V = self.post_ffwd_norm(V)

        return V

class Encoder(nn.Module):

    def __init__(self, embed_input_dim, embed_nlayers, embed_dim, mlp_input_dim, mlp_nlayers, mlp_dim, attn_blocks_n, attn_block_num_heads, attn_block_ffwd_on, attn_block_ffwd_nlayers, attn_block_ffwd_dim, gumbel_softmax_config, out_dim, doWij, ae_dim, ae_depth, do_gumbel, mass_scale):

        super().__init__()

        # number of target (T) objects (g1,g2,ISR)
        self.T = out_dim

        # embed, In -> Out : J,C -> J,E
        self.embed = DNN_block([embed_input_dim] + (embed_nlayers+1)*[embed_dim], normalize_input=True)

        # position encoding based on (eta,cos(phi),sin(phi))
        self.doWij = doWij
        if self.doWij:
            # MLP for Wij
            mlp_dimensions = [mlp_input_dim] + mlp_nlayers*[mlp_dim] + [1]
            self.mlp = DNN_block(mlp_dimensions, normalize_input=False)

        # object attention blocks, In -> Out : J,E -> J,E
        ffwd_dims = [[attn_block_ffwd_dim]*attn_block_ffwd_nlayers + [embed_dim]] * attn_blocks_n
        self.obj_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
        # candidate attention blocks, In -> Out : T,E -> T,E
        self.gumbel_softmax_config = gumbel_softmax_config
        self.cand_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
        # cross attention blocks, In -> Out : J,E -> J,E
        self.cross_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])

        # build candidates from T scores: J,E -> T,E
        
        # autoencoder blocks E-T+1 -> B -> E-T+1
        # drops the T jet scores from the features and adds the mass
        self.ae_in = DNN_block(cascade_dims(embed_dim-self.T+1, ae_dim, ae_depth), normalize_input=False)
        self.ae_out = DNN_block(cascade_dims(ae_dim, embed_dim-self.T+1, ae_depth), normalize_input=False)

        self.do_gumbel = do_gumbel
        self.mass_scale = mass_scale

    def forward(self, x, w, mask, loss=None):

        # embed and remask, In -> Out : J,C -> J,E
        originalx = x
        x = self.embed(x)
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

        # attention mechanism
        for ib in range(len(self.obj_blocks)):

            # pairwise + MLP
            if self.doWij:
                if ib == 0:
                    wij = pairwise(w)
                    wij = self.mlp(wij)
                    wij = wij.squeeze(-1)
                    wij = wij.repeat(self.obj_blocks[ib].attn.num_heads,1,1)
            else:
                wij = None


            # apply attention and remask, In -> Out : J,C -> J,E
            x = self.obj_blocks[ib](Q=x, K=x, V=x, key_padding_mask=mask.bool(), attn_mask=wij)
            x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

            # candidate attention
            jet_choice  = self.get_jet_choice(x)
            c = torch.bmm(jet_choice.transpose(2,1), x) # (T,J)x (J,E) -> T,E
            c = self.cand_blocks[ib](Q=c, K=c, V=c, key_padding_mask=None, attn_mask=None) # T,E

            # cross attention, In -> Out : (J,E)x(E,T)x(T,E) -> J,E
            x = self.cross_blocks[ib](Q=x, K=c, V=c, key_padding_mask=None, attn_mask=None) # J,E
            x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)
            
        #build candidate mass from original jet 4-vector
        jp4 = x_to_p4(originalx)
        jet_choice  = self.get_jet_choice(x)
        candidates_p4 = torch.bmm(jet_choice.transpose(2,1), jp4)
        cmass = ms_from_p4s(candidates_p4)/self.mass_scale #arbitrary scaling factor

        c       = torch.cat([c[:,:,self.T:],cmass[:,:,None]],-1) #drop the category scores, add the mass

        #autoencoders
        c0        = c[:,0]
        c0_latent = self.ae_in(c0)
        c0_out    = self.ae_out(c0_latent)

        c1        = c[:,1]
        c1_latent = self.ae_in(c1)
        c1_out    = self.ae_out(c1_latent)

        loss  = get_mse(c0, c0_out) + get_mse(c1, c1_out)
        xloss = get_mse(c0, c1_out) + get_mse(c1, c0_out)

        return loss, xloss, candidates_p4, jet_choice

    def get_jet_choice(self,x):
        if self.T==3:
            # after initialization the jet scores for the 3 categories are similar and
            # often it collapses to predict everything ISR. By shifting before the softmax
            # it starts predicting jets in BSM candidates, and later adjusts ISR predictions
            x[:,:,2] -= 1

        if self.training:
            # differential but relies on probability distribution
            if self.do_gumbel:
                jet_choice = nn.functional.gumbel_softmax(x[:,:,:self.T], dim=2, **self.gumbel_softmax_config) # J, T
            else:
                # divide by some temperature to make the preditions almost one-hot
                jet_choice = nn.functional.softmax(x[:,:,:self.T]/0.1, dim=2) # J, T
        else:
            # not differential but justs max per row
            jet_choice = nn.functional.one_hot(torch.argmax(x[:,:,:self.T],dim=-1), num_classes=self.T).float() # J, T
        return jet_choice

def x_to_p4(x):
    pt = torch.exp(x[..., 0])
    pt[pt==1] = 0 #padded log(pt) had been set -inf->0
    eta = x[..., 1]
    px = pt*x[..., 2]
    py = pt*x[..., 3]
    e = torch.exp(x[..., 4])
    e[e==1] = 0 #padded log(e) had been set -inf->0
    pz = pt * torch.sinh(eta)

    return torch.stack([e,px,py,pz], -1)
    
def m2s_from_p4s(p4s, eps=1e-4):
    #add some tiny energy to prevent negative masses from rounding errors
    m2s = (p4s[...,0]*(1+eps))**2 - p4s[...,1]**2 - p4s[...,2]**2 - p4s[...,3]**2+eps
    return m2s

def ms_from_p4s(p4s, eps=1e-4):
    m2s = m2s_from_p4s(p4s, eps=eps)
    return torch.sqrt(m2s)

def cascade_dims(input_dim, output_dim, depth):
    dimensions = [int(output_dim + (input_dim - output_dim)*(depth-i)/depth) for i in range(depth+1)]
    return dimensions

def get_mse(xin, xout):
    return torch.mean((xin-xout)**2, -1)
