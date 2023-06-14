import torch
import torch.nn as nn

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

class Encoder(nn.Module):

    def __init__(self, embed_input_dim, embed_nlayers, embed_dim, mlp_input_dim, mlp_nlayers, mlp_dim, attn_blocks_n, attn_block_num_heads, attn_block_ffwd_on, attn_block_ffwd_nlayers, attn_block_ffwd_dim, gumble_softmax_config, out_dim, doWij, doCandidateAttention):

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
        
        # candidate attention blocks
        self.doCandidateAttention = doCandidateAttention
        if self.doCandidateAttention:

            # candidate attention blocks, In -> Out : T,E -> T,E
            self.gumble_softmax_config = gumble_softmax_config
            self.cand_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
            # cross attention blocks, In -> Out : J,E -> J,E
            self.cross_blocks = nn.ModuleList([AttnBlock(embed_dim=embed_dim, num_heads=attn_block_num_heads, ffwd_dims=ffwd_dims[cfg]) for cfg in range(attn_blocks_n)])
        
        # final head, In -> Out : J,E -> J,T
        self.final = Embed(embed_dim, embed_nlayers*[embed_dim], normalize_input=True, final_layer=[embed_dim,self.T])

    def forward(self, x, w, mask, loss=None):

        # embed and remask, In -> Out : J,C -> J,E
        x = self.embed(x)
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

        # cumulative loss
        self.block_mass_loss = 0

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
            if self.doCandidateAttention:

                if self.training:
                    # differential but relies on probability distribution
                    c = nn.functional.gumbel_softmax(x[:,:,:self.T], dim=2, **self.gumble_softmax_config) # J, T
                else:
                    # not differential but justs max per row
                    c = nn.functional.one_hot(torch.argmax(x[:,:,:self.T],dim=-1), num_classes=self.T).float() # J, T
                c = torch.bmm(c.transpose(2,1), x) # (T,J)x (J,E) -> T,E
                c = self.cand_blocks[ib](Q=c, K=c, V=c, key_padding_mask=None, attn_mask=None) # T,E
            
                # cross attention, In -> Out : (J,E)x(E,T)x(T,E) -> J,E
                x = self.cross_blocks[ib](Q=x, K=c, V=c, key_padding_mask=None, attn_mask=None) # J,E
                x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)
        
        # final head J,E -> J,T
        x = self.final(x)
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

        return x, w
