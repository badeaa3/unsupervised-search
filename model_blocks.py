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

class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, final_layer=None):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            input_dim = dim
        if final_layer is not None:
            module_list.append(nn.Linear(final_layer[0], final_layer[1]))
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            x = self.input_bn(x.transpose(1,2)) # must transpose because batchnorm expects N,C,L rather than N,L,C like multihead
            x = x.transpose(1,2) # revert transpose
        x = self.embed(x)
        return x

class AttnBlock(nn.Module):

    def __init__(self,
                 embed_dim = 4,
                 num_heads = 1,
                 attn_dropout = 0,
                 add_bias_kv = False,
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

class AE_block(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4):

        super().__init__()
        dimensions = [int(output_dim + (input_dim - output_dim)*(depth-i)/depth) for i in range(depth+1)]
        layers = []
        for dim_in, dim_out in zip(dimensions, dimensions[1:]):
            if dim_out != output_dim:
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
        return self.net(x)

class Encoder(nn.Module):

    def __init__(self, embed_input_dim, embed_nlayers, embed_dim, mlp_input_dim, mlp_nlayers, mlp_dim, attn_blocks_n, attn_block_num_heads, attn_block_ffwd_on, attn_block_ffwd_nlayers, attn_block_ffwd_dim, gumble_softmax_config, out_dim, doWij, doCandidateAttention, ae_dim, ae_depth):

        super().__init__()

        # number of target (T) objects (g1,g2,ISR)
        self.T = 3

        # embed, In -> Out : J,C -> J,E
        self.embed = Embed(embed_input_dim, embed_nlayers*[embed_dim], normalize_input=True)

        # position encoding based on (eta,cos(phi),sin(phi))
        self.doWij = doWij
        if self.doWij:
            # MLP for Wij
            self.mlp = Embed(mlp_input_dim, mlp_nlayers*[mlp_dim], normalize_input=False, final_layer=[mlp_dim,1])

        # object attention blocks, In -> Out : J,E -> J,E
        ffwd_dims = [[attn_block_ffwd_dim]*attn_block_ffwd_nlayers + [embed_dim]] * attn_blocks_n
        self.obj_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
        # candidate attention blocks, In -> Out : T,E -> T,E
        self.gumble_softmax_config = gumble_softmax_config
        self.cand_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
        # cross attention blocks, In -> Out : J,E -> J,E
        self.cross_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
        # ends in a candidate attention block
        # final output is T,E
        self.ae_in  = AE_block(embed_dim+1, ae_dim, ae_depth)
        self.ae_out = AE_block(ae_dim+1, embed_dim, ae_depth)

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
            x = self.obj_blocks[ib](Q=x, K=x, V=x, key_padding_mask=mask.bool(), attn_mask=wij) #.repeat(self.obj_blocks[ib].attn.num_heads,1,1))
            x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

            # candidate attention
            if self.training:
                # differential but relies on probability distribution
                cchoice = nn.functional.gumbel_softmax(x[:,:,:self.T], dim=2, **self.gumble_softmax_config) # J, T
            else:
                # not differential but justs max per row
                cchoice = nn.functional.one_hot(torch.argmax(x[:,:,:self.T],dim=-1), num_classes=self.T).float() # J, T
            c = torch.bmm(cchoice.transpose(2,1), x) # (T,J)x (J,E) -> T,E
            #FIXME should we enter the AE before or after the candidate self-attention?
            c = self.cand_blocks[ib](Q=c, K=c, V=c, key_padding_mask=None, attn_mask=None) # T,E
            
            if ib==len(self.obj_blocks)-1: #incomplete last block
                #build candidate mass from original jet 4-vector
                jp4 = x_to_p4(originalx) # J, 4
                cp4 = torch.bmm(cchoice.transpose(2,1), jp4) # (T,J)x (J,4) -> T,4
                cmass = ms_from_p4s(cp4) # T

                #build random candidates
                randomchoice = torch.from_numpy(np.zeros(cchoice.shape)).float().to("cuda:0")
                randomindices = np.random.randint(self.T, size=randomchoice.shape[:-1])
                randomindices[mask.cpu()] = 0
                np.put_along_axis(randomchoice,randomindices[:,:,np.newaxis],True,axis=-1)
                randomchoice = torch.tensor(randomchoice, requires_grad=False)
                crandom = torch.bmm(randomchoice.transpose(2,1), x)
                crandom = self.cand_blocks[ib](Q=c, K=c, V=c, key_padding_mask=None, attn_mask=None)
                crandomp4 = torch.bmm(randomchoice.transpose(2,1), jp4)
                crandommass = ms_from_p4s(crandomp4)
            else:
                # cross attention, In -> Out : (J,E)x(E,T)x(T,E) -> J,E
                x = self.cross_blocks[ib](Q=x, K=c, V=c, key_padding_mask=None, attn_mask=None) # J,E
                x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

        #autoencoders
        cISR = c[:,0]

        c1        = c[:,1]
        c1mass    = cmass[:,1]
        c1_latent = self.ae_in(torch.cat([c1,c1mass[:,None]],-1))
        c1_out    = self.ae_out(torch.cat([c1_latent,c1mass[:,None]],-1))

        c2        = c[:,2]
        c2mass    = cmass[:,2]
        c2_latent = self.ae_in(torch.cat([c2,c2mass[:,None]],-1))
        c2_out    = self.ae_out(torch.cat([c2_latent,c2mass[:,None]],-1))

        c1random        = crandom[:,1]
        c1randommass    = crandommass[:,1]
        c1random_latent = self.ae_in(torch.cat([c1random,c1randommass[:,None]],-1))
        c1random_out    = self.ae_out(torch.cat([c1random_latent,c1randommass[:,None]],-1))

        c2random        = crandom[:,2]
        c2randommass    = crandommass[:,2]
        c2random_latent = self.ae_in(torch.cat([c2random,c2randommass[:,None]],-1))
        c2random_out    = self.ae_out(torch.cat([c2random_latent,c2randommass[:,None]],-1))
        #inspect(c,cmass,crandom,crandommass, c1_out,c2_out,c1random_out,c2random_out, x, originalx, jp4, cp4, cchoice, randomchoice)

        return c1, c2, c1_out, c2_out, c1random_out, c2random_out

def inspect(c,cmass,crandom,crandommass, c1_out,c2_out,c1random_out,c2random_out, x, originalx, jp4, cp4, cchoice, randomchoice):
        print("Nans", torch.isnan(c).sum(), torch.isnan(cmass).sum(),torch.isnan(crandom).sum(),torch.isnan(crandommass).sum(),  torch.isnan(torch.stack([c1_out,c2_out,c1random_out,c2random_out])).sum())
        print("Input", x[0], originalx[0], jp4[0])
        print("Candidates", c[0], cchoice[0], cp4[0], cmass[0])
        print("Random candidates", crandom[0], randomchoice[0], crandommass[0])
        print("Candidates out", c1_out[0], c2_out[0], c1random_out[0], c2random_out[0])

def x_to_p4(x):
    pt = torch.exp(x[..., 0])
    pt[pt==1] = 0
    eta = x[..., 1]
    px = pt*x[..., 2]
    py = pt*x[..., 3]
    e = torch.exp(x[..., 4])
    e[e==1] = 0
    pz = pt * torch.sinh(eta)

    return torch.stack([e,px,py,pz], -1)
    
def ms_from_p4s(p4s):
    ''' copied from energyflow '''
    m2s = p4s[...,0]**2 - p4s[...,1]**2 - p4s[...,2]**2 - p4s[...,3]**2
    return torch.sign(m2s)*torch.sqrt(torch.abs(m2s))
