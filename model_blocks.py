import torch
import torch.nn as nn
import numpy as np
from info_nce import InfoNCE
#import energyflow_torch as ef_torch
#efset = ef_torch.EFPSet('d<=3')

device = "cuda:0"
device = "cpu"

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
        self.input_ln = nn.LayerNorm(input_dim) if normalize_input else None

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
            if len(x.shape) ==2:
            	#x = self.input_bn(x)
            	x = self.input_ln(x)
            elif len(x.shape) >=3:
            	x = self.input_bn(x.transpose(1,2)) # must transpose because batchnorm expects N,C,L rather than N,L,C like multihead
            	x = x.transpose(1,2) # revert transpose
        return self.net(x)

class AttnBlock(nn.Module):

    def __init__(self,
                 embed_dim = 4,
                 num_heads = 1,
                 attn_dropout = 0,
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
            if i != len(ffwd_dims)-1:
                self.ffwd.extend([
                    nn.Linear(input_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                ])
            else:
                self.ffwd.extend([
                    nn.Linear(input_dim, dim),
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

    def __init__(self, embed_input_dim, embed_nlayers, embed_dim, mlp_input_dim, mlp_nlayers, mlp_dim, attn_blocks_n, attn_block_num_heads, attn_block_ffwd_on, attn_block_ffwd_nlayers, attn_block_ffwd_dim, gumbel_softmax_config, out_dim, doWij, ae_dim, ae_depth, do_gumbel, mass_scale, do_vae=False, add_mass_feature=True, add_mass_latent=False, sync_rand=False, over_jet_count=True, random_mode=False, rand_cross_candidates=True, remove_mass_from_loss=False):

        super().__init__()

        # number of target (T) objects (g1,g2,ISR)
        self.T = out_dim
        if not hasattr(self.T, '__iter__'):
            self.T = [self.T]*attn_blocks_n
        else:
            assert len(self.T) == attn_blocks_n
        self.maxT = max(self.T)

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
        self.ae_in = DNN_block(cascade_dims(embed_dim-self.maxT+(1 if add_mass_feature else 0), ae_dim*(2 if do_vae else 1), ae_depth), normalize_input=False)
        self.ae_out = DNN_block(cascade_dims(ae_dim+(1 if add_mass_latent else 0), embed_dim-self.maxT++(1 if add_mass_feature else 0), ae_depth), normalize_input=False)
        self.projector = DNN_block([ae_dim]*2, normalize_input=False)
        #self.ae_in = DNN_block(cascade_dims(4+1, ae_dim*(2 if do_vae else 1), ae_depth), normalize_input=False)
        #self.ae_out = DNN_block(cascade_dims(ae_dim+(1 if add_mass_latent else 0), 4+1, ae_depth), normalize_input=False)
        #self.ae_in = DNN_block(cascade_dims(13+1, ae_dim*2, ae_depth), normalize_input=False)
        #self.ae_out = DNN_block(cascade_dims(ae_dim+1, 13+1, ae_depth), normalize_input=False)

        self.do_gumbel = do_gumbel
        self.do_vae = do_vae
        self.mass_scale = mass_scale
        self.add_mass_feature = add_mass_feature
        self.add_mass_latent = add_mass_latent
        self.sync_rand = sync_rand
        self.over_jet_count = over_jet_count
        self.random_mode = random_mode
        self.rand_cross_candidates = rand_cross_candidates
        self.remove_mass_from_loss = remove_mass_from_loss
        self.infoNCE = InfoNCE(negative_mode='paired')


    def reparameterize(self, x1, x2, sync=False):
        split_size = x1.shape[-1]//2
        mu1 = x1[...,:split_size]
        logvar1 = x1[...,split_size:]
        std1 = torch.exp(0.5 * logvar1)
        eps1 = torch.randn_like(std1)

        mu2 = x2[...,:split_size]
        logvar2 = x2[...,split_size:]
        std2 = torch.exp(0.5 * logvar2)
        if sync:
            eps2 = eps1
        else:
            eps2 = torch.randn_like(std2)
        return (eps1 * std1 + mu1, eps2 * std2 +mu2)

    def forward(self, x, w, mask, tau):

        # embed and remask, In -> Out : J,C -> J,E
        originalx = x
        x = self.embed(x)
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

        masses  = []
        mu_logvar  = []
        loss  = []
        loss_crossed = []
        latent_mse = []
        loss_infoNCE = []
        rand_crossed = []
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
            c = self.build_candidates(x, originalx, mask, self.T[ib], simple=True)
            c = self.cand_blocks[ib](Q=c, K=c, V=c, key_padding_mask=None, attn_mask=None) # T,E

            # cross attention, In -> Out : (J,E)x(E,T)x(T,E) -> J,E
            x = self.cross_blocks[ib](Q=x, K=c, V=c, key_padding_mask=None, attn_mask=None) # J,E
            x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)

            damp_factor = 1 #len(self.obj_blocks)-ib
            #build candidate mass from original jet 4-vector
            c, candidates_p4, jet_choice, cmass = self.build_candidates(x, originalx, mask, self.T[ib])
            rand_c, rand_candidates_p4, rand_jet_choice, rand_cmass = self.build_candidates(x, originalx, mask, self.T[ib], random=self.random_mode)

            #autoencoders
            c0        = c[:,0]
            c0_latent = self.ae_in(c0)
            c1        = c[:,1]
            c1_latent = self.ae_in(c1)

            if self.do_vae:
                c0_latent_rep, c1_latent_rep = self.reparameterize(c0_latent, c1_latent, self.sync_rand)
            else:
                c0_latent_rep, c1_latent_rep = c0_latent, c1_latent
            if self.add_mass_latent:
                c0_latent_rep = torch.cat([c0_latent_rep, cmass[:,0,None]],-1)
                c1_latent_rep = torch.cat([c1_latent_rep, cmass[:,1,None]],-1)

            c0_out    = self.ae_out(c0_latent_rep)
            c1_out    = self.ae_out(c1_latent_rep)

            #autoencoders
            rand_c0        = rand_c[:,0]
            rand_c0_latent = self.ae_in(rand_c0)
            rand_c1        = rand_c[:,1]
            rand_c1_latent = self.ae_in(rand_c1)

            if self.do_vae:
                rand_c0_latent_rep, rand_c1_latent_rep = self.reparameterize(rand_c0_latent, rand_c1_latent, self.sync_rand)
            else:
                rand_c0_latent_rep, rand_c1_latent_rep = rand_c0_latent, rand_c1_latent
            if self.add_mass_latent:
                rand_c0_latent_rep = torch.cat([rand_c0_latent_rep, rand_cmass[:,0,None]],-1)
                rand_c1_latent_rep = torch.cat([rand_c1_latent_rep, rand_cmass[:,1,None]],-1)

            projected_c0_latent_rep = self.projector(c0_latent_rep)
            projected_c1_latent_rep = self.projector(c1_latent_rep)
            projected_rand_c0_latent_rep = self.projector(rand_c0_latent_rep)
            projected_rand_c1_latent_rep = self.projector(rand_c1_latent_rep)

            rand_c0_out    = self.ae_out(rand_c0_latent_rep)
            rand_c1_out    = self.ae_out(rand_c1_latent_rep)

            mu_logvar.append(torch.stack((c0_latent, c1_latent),dim=1))
            loss.append( get_mse(c0, c0_out, self.remove_mass_from_loss) + get_mse(c1, c1_out, self.remove_mass_from_loss)) #FIXME
            loss_crossed.append( (get_mse(c0, c1_out) + get_mse(c1, c0_out) )/damp_factor)
            latent_mse.append( get_mse(c0_latent_rep, c1_latent_rep)/damp_factor )
            #loss_infoNCE.append( self.infoNCE(c0_latent_rep, c1_latent_rep, rand_c0_latent_rep[:,None,:])+self.infoNCE(c1_latent_rep, c0_latent_rep, rand_c1_latent_rep[:,None,:]) )
            loss_infoNCE.append( self.infoNCE(projected_c0_latent_rep, projected_c1_latent_rep, projected_rand_c0_latent_rep[:,None,:])+self.infoNCE(projected_c1_latent_rep, projected_c0_latent_rep, projected_rand_c1_latent_rep[:,None,:]) )
            if self.rand_cross_candidates:
                rand_crossed.append( get_mse(rand_c0_latent_rep, rand_c1_latent_rep)/damp_factor )
            else:
                rand_crossed.append( (get_mse(c0_latent_rep,rand_c0_latent_rep) + get_mse(c1_latent_rep, rand_c1_latent_rep))/damp_factor )
            masses.append( ms_from_p4s(candidates_p4) )

        loss_crossed = torch.stack(loss_crossed , -1)
        loss  = torch.stack(loss , -1)
        mu_logvar  = torch.stack(mu_logvar , -1)
        masses  = torch.stack(masses , 2)
        loss_infoNCE = torch.stack(loss_infoNCE , -1)
        latent_mse = torch.stack(latent_mse , -1)
        rand_crossed = torch.stack(rand_crossed , -1)
        
        return loss, mu_logvar, latent_mse, rand_crossed, candidates_p4, jet_choice, (masses, c0, c1, c0_latent_rep, c1_latent_rep, c0_out, c1_out)
        #Use InfoNCE loss
        #return loss, mu_logvar, loss_infoNCE, 0, candidates_p4, jet_choice, (masses, c0, c1, c0_latent_rep, c1_latent_rep, c0_out, c1_out)

    def build_candidates(self, x, originalx, mask, thisT, simple=False, addefc=False, addefp=False, random=False):
        #build candidate mass from original jet 4-vector
        jet_choice  = self.get_jet_choice(x, thisT)
        if random:
            jet_choice = self.get_random_choice(jet_choice, mask, thisT, random)
        jet_choice  = jet_choice.masked_fill(mask.unsqueeze(-1).repeat(1,1,jet_choice.shape[-1]).bool(), 0)
        c = torch.bmm(jet_choice.transpose(2,1), x) # (T,J)x (J,E) -> T,E
        jet_count = torch.unsqueeze(torch.sum(jet_choice, -2)+1e-3,-1)
        if self.over_jet_count:
            c = c/jet_count
        if simple: return c

        jp4 = x_to_p4(originalx)
        candidates_p4 = torch.bmm(jet_choice.transpose(2,1), jp4)
        candidates_jet_p4 = torch.mul(jet_choice.transpose(2,1)[...,None], jp4[:,None])

        #cpt   = pts_from_p4s(candidates_p4)/self.mass_scale #arbitrary scaling factor
        cmass = ms_from_p4s(candidates_p4)
        cmass = cmass/self.mass_scale #arbitrary scaling factor
        #cmass = torch.sqrt(cmass)-20
        #cmass = torch.maximum(cmass, torch.full_like(cmass, 300))
        #cmass = torch.log(cmass)

        c       = c[:,:,self.maxT:] #drop the category scores
        if self.add_mass_feature:
            c       = torch.cat([c, cmass[:,:,None]],-1) #add the mass
        if addefc:
            candidates_efc = []
            for i in range(self.maxT):
                c_count = jet_count[:,i].squeeze()
                pt = pts_from_p4s(candidates_jet_p4[:,i])
                dR2 = dr2s_from_p4s(candidates_jet_p4[:,i])
                c_efcs = []
                for beta in (2,):
                #for beta in (0.2,1,2):
                    efcs = {0:1}
                    MAX = 5
                    for c in range(1,MAX+1):
                         efcs[c] = EFC(pt, dR2, c, beta) #compute only once the EFC to be fast
                    for c in range(1,MAX):
                         c_efc = torch.where(c<c_count, efcs[c+1]*efcs[c-1]/torch.square(efcs[c]), 0) #compute myself C
                         c_efcs.append(c_efc)
                candidates_efc.append( torch.stack(c_efcs,-1) )
            candidates_efc = torch.stack(candidates_efc,1)
            c = torch.cat([candidates_efc, cmass[:,:,None]],-1) #not using the original candidate features, only EFC+mass
            #print("jet_count",jet_count[0].squeeze())
            #print("c",c[0])
        if addefp:
            candidates_efp = []
            for i in range(self.maxT):
                efp = efset.batch_compute(candidates_jet_p4[:,i], n_jobs=1)
                print("i",i, efp.shape)
                candidates_efp.append(efp)
            candidates_efp  = torch.stack(candidates_efp,1)
            print("shapes",candidates_efp.shape, c.shape, candidates_jet_p4.shape)
#shapes torch.Size([2048, 2, 13]) torch.Size([2048, 2, 15]) torch.Size([2048, 2, 12, 4])
            c       = torch.cat([candidates_efp, cmass[:,:,None]],-1) #add the mass
            print(c.dtype, candidates_efp.dtype, cmass.dtype)
        return c, candidates_p4, jet_choice, cmass

    def get_jet_choice(self,x,thisT,tau=1):
        if self.maxT==4:
            # after initialization the jet scores for the 3 categories are similar and
            # often it collapses to predict everything ISR. By shifting before the softmax
            # it starts predicting jets in BSM candidates, and later adjusts ISR predictions
            x[:,:,2] -= 1

        if self.training:
            # differential but relies on probability distribution
            if self.do_gumbel:
                jet_choice = nn.functional.gumbel_softmax(x[:,:,:thisT]/tau, dim=2, **self.gumbel_softmax_config) # J, T
            else:
                # divide by some temperature to make the predictions almost one-hot
                jet_choice = nn.functional.softmax(x[:,:,:thisT]/tau, dim=2) # J, T
            if thisT < self.maxT:
                jet_choice = torch.cat([jet_choice, torch.zeros(jet_choice.shape[:2]+(1,)).to('cuda:0')],-1)
        else:
            # not differential but justs max per row
            # if thisT < self.maxT the third column will be automatically zeros
            jet_choice = nn.functional.one_hot(torch.argmax(x[:,:,:thisT],dim=-1), num_classes=self.maxT).float() # J, T
        return jet_choice

    def get_random_choice(self, cchoice, mask, thisT, random_mode):
        nbatch, njet = cchoice.shape[:2]
        cut = 6

        if random_mode == 'shuffle6':
            # reshuffle the 6 leading predictions
            indices = torch.argsort(torch.rand((nbatch, cut)), dim=-1)
            indices = torch.cat([indices, torch.arange(cut,njet).repeat(nbatch,1)],-1).type(torch.int64).to(device)
            indices = indices[:,:,None].repeat(1,1,thisT)
            randomchoice = torch.gather(cchoice, dim=1, index=indices)
            return randomchoice

        if random_mode == 'reverse_scores':
            # resverse some of the scores
            indices = torch.argsort(torch.rand((nbatch, cut, thisT)), dim=-1)
            indices = torch.cat([indices, torch.arange(thisT).repeat(nbatch,njet-cut,1)],1).type(torch.int64).to(device)
            randomchoice = torch.gather(cchoice, dim=2, index=indices)
            return randomchoice

        if random_mode == 'reverse_both':
            rowindices = torch.argsort(torch.rand((nbatch, cut)), dim=-1)
            rowindices = torch.cat([rowindices, torch.arange(cut,njet).repeat(nbatch,1)],-1).type(torch.int64).to(device)
            rowindices = rowindices[:,:,None].repeat(1,1,thisT)
            randomchoice = torch.gather(cchoice, dim=1, index=rowindices)
            scoreindices = torch.argsort(torch.rand((nbatch, cut, thisT)), dim=-1)
            scoreindices = torch.cat([scoreindices, torch.arange(thisT).repeat(nbatch,njet-cut,1)],1).type(torch.int64).to(device)
            randomchoice = torch.gather(randomchoice, dim=2, index=scoreindices)
            return randomchoice
        print("random_mode not known",random_mode)


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

def dr2s_from_p4s(p4s):
    from math import pi
    eps = 1e-3
    eta = torch.where(p4s[...,0] > 0, torch.arctanh(p4s[...,3]/(p4s[...,0]+eps)), 0)
    phi = torch.where(p4s[...,0] > 0, torch.arctan2(p4s[...,2]+eps, p4s[...,1]+eps), 0)
    deta = eta[...,:,np.newaxis]-eta[...,np.newaxis,:]
    dphi = torch.abs(phi[...,:,np.newaxis]-phi[...,np.newaxis,:])
    dphi = torch.where(dphi > pi, 2*pi - dphi, dphi)
    dR2 = torch.square(deta)+torch.square(dphi)
    #dR2 = torch.square(deta)+torch.square(dphi)
#    dR2 = torch.nan_to_num(torch.square(deta)+torch.square(dphi))
    #print("jp4",p4s[0])
    #print("eta",eta[0])
    #print("deta",deta[0])
    #print("phi",phi[0])
    #print("dphi",dphi[0])
    #print("dR2",dR2[0])
    return dR2

def pts_from_p4s(p4s, eps=1e-4):
    return torch.sqrt(p4s[...,1]**2+p4s[...,2]**2+eps)

def ms_from_p4s(p4s, eps=1e-4):
    m2s = m2s_from_p4s(p4s, eps=eps)
    return torch.sqrt(m2s)

def cascade_dims(input_dim, output_dim, depth):
    dimensions = [int(output_dim + (input_dim - output_dim)*(depth-i)/depth) for i in range(depth+1)]
    return dimensions

def get_mse(xin, xout, remove_mass_from_loss=False):
    if remove_mass_from_loss:
        return torch.mean((xin[...,:-1]-xout[...,:-1])**2, -1)
    return torch.mean((xin-xout)**2, -1)

#https://arxiv.org/pdf/1305.0007.pdf
ptcombs = {}
drcombs = {}
def fill_combs_EFC(jpt):
    MAXEFC = 7
    for n in range(2,MAXEFC):
        if n > jpt.shape[1]: break
        with torch.no_grad():
            indices = torch.arange(jpt.shape[1]).to('cuda:0')
            ptcombs[n] = torch.combinations(indices, n)
            for i, comb in enumerate(ptcombs[n]):
                combout = torch.combinations(comb, 2)
                if not n in drcombs:
                    drcombs[n] = torch.empty((ptcombs[n].shape[0], combout.shape[0], 2),dtype=torch.int64).to('cuda:0')
                drcombs[n][i] = combout

def EFC(pt, dR2, n,beta=2):
    if not ptcombs:
        fill_combs_EFC(pt)
    if n==0: return 1
    if n==1: return torch.sum(pt,dim=-1)
    drcomb = drcombs[n]
    P = n
    N, J = pt.shape
    A, B, _  = drcomb.shape
    #N events, J jets per event, P jets to pick, A combinations of P jets, B pairs out of P
    #ptcombs = (A, P)
    #drcombs = (A, B, 2)
    #dR2 = (N,J,J)
    indexrow = drcomb[None,:,:,1,None,None]
    indexrow = indexrow.expand(N,-1,-1,J,-1)
    indexcol = drcomb[None,:,:,0,None]
    indexcol = indexcol.expand(N,-1,-1,-1)
    dR2expand = dR2[:,None,None,:,:]
    dR2expand = dR2expand.expand(-1,A,B,-1,-1)

    row = torch.gather(dR2expand, dim=-1, index=indexrow).squeeze(-1)
    drs = torch.gather(row, dim=-1, index=indexcol).squeeze(-1)
    
    ptexpand = pt[:,None,None,:]
    ptexpand = ptexpand.expand(-1,A,P,-1)
    indexpt = ptcombs[n]
    indexpt = indexpt[None,:,:,None]
    indexpt = indexpt.expand(N,-1,-1,-1)

    pts = torch.gather(ptexpand, dim=-1, index=indexpt).squeeze(-1)
    
    #print("----")
    #print(drs[0])
    #print(drs.shape)
    drs = torch.prod(drs,dim=-1)
    if beta!=2:
        drs = torch.where(drs>0, torch.pow(drs, beta/2), 0)
    pts = torch.prod(pts,dim=-1)
    terms = pts*drs
    efc = torch.sum(terms,dim=-1)
    return efc

def r_EFC(jpt, jdr2, n, beta=2):
    return EFC(jpt, jdr2, n+1,beta)/EFC(jpt, jdr2, n)

def C_EFC(jpt, jdr2, n, beta=2):
    return r_EFC(jpt, jdr2, n, beta)/r_EFC(jpt, jdr2, n-1, beta)

def jp4_to_ptdr2(jp4):
    pt = pts_from_p4s(jp4)
