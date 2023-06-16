import torch, numpy as np

def get_mass_max(x, p):

    # indices and set pad to ISR
    ap = p.argmax(2)
    ap[p.sum(-1)==0] = 2
    a = np.eye(p.shape[2])[ap]
    mom = np.matmul(a.transpose(0,2,1),x) # basis is (E,px,py,pz)
    # convert to pt, eta, phi, m
    pt = np.sqrt(mom[:,:,1]**2 + mom[:,:,2]**2) # pt = sqrt(px^2 + py^2)
    eta = np.arcsinh(mom[:,:,3]/pt) # pz = pt sinh(eta)
    phi = np.arctan2(mom[:,:,2],mom[:,:,1]) # py/px = tan(phi)
    m = np.sqrt(mom[:,:,0]**2 - mom[:,:,1]**2 - mom[:,:,2]**2 - mom[:,:,3]**2) # e^2 = m^2 + p^2
    mom = torch.stack([pt,eta,phi,m],-1)
    # fix the padded to be ap=-1
    ap[x[:,:,0]==0]=-1
    return mom, ap

def get_mass_set(X,P):

    is_lead_P = P[:,:,0]
    is_sublead_P = P[:,:,1]
    is_ISR_P = P[:,:,2]

    emptyjets =  X[:,:,3]==0
    is_ISR_P[emptyjets] = 99 #in-place change prevents gradients, can not be used for smooth
    
    jet_4p = X
    
    n_event = jet_4p.shape[0]
    n_jet = 8
    n_ISR = n_jet-6
    n_gluino = 3

    # find the ISR jets
    # Exclude jets with highest ISR score until there are 6 gluino jets left
    v, i = torch.sort(is_ISR_P, dim=1, descending=True)
    ISR_threshold = v[:,1]
    ISR_mask = (is_ISR_P >= ISR_threshold[:,None]).int()
    
    #zero out ISR jets
    renorm_lead_P = is_lead_P
    renorm_lead_P = renorm_lead_P*(1-ISR_mask)
    v, i = torch.sort(renorm_lead_P, dim=1, descending=True)

    lead_threshold = v[:,2]
    lead_mask = (renorm_lead_P >= lead_threshold[:,None]).int()
    sublead_mask = 1 - lead_mask - ISR_mask
    
    a = torch.stack([lead_mask,sublead_mask,ISR_mask],1)
    mom = np.matmul(a,X)
    pt = np.sqrt(mom[:,:,1]**2 + mom[:,:,2]**2) # pt = sqrt(px^2 + py^2)
    eta = np.arcsinh(mom[:,:,3]/pt) # pz = pt sinh(eta)
    phi = np.arctan2(mom[:,:,2],mom[:,:,1]) # py/px = tan(phi)
    m = np.sqrt(mom[:,:,0]**2 - mom[:,:,1]**2 - mom[:,:,2]**2 - mom[:,:,3]**2) # e^2 = m^2 + p^2
    mom = torch.stack([pt,eta,phi,m],-1)
    
    # get prediction labels
    ap = a.argmax(1)
    # fix the padded to be ap=-1
    ap[X[:,:,0]==0]=-1

    return mom,ap

if __name__ == "__main__":

    x = torch.Tensor([
        [ 1.6803, -6.7709, -0.1952],
        [-0.7437, -0.4924, -1.3779],
        [-0.6305, -0.5535, -1.4105],
        [-0.8678, -0.4070, -1.3154],
        [-0.9150, -0.4179, -1.1902],
        [-0.8382, -0.5540, -1.0531],
        [-0.9211, -0.5754, -0.8581],
        [-0.9274, -0.7717, -0.4650]])
    x = torch.Tensor([[1., 0., 0.],
                      [0., 1., 0.],
                      [1., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 1.]])
    P = torch.unsqueeze(x,0).repeat(6,1,1)
    P = torch.nn.Softmax(2)(P)
    X = torch.randn(P.shape[0],P.shape[1],4)

    get_massFix(X,P)
    
