'''
batch manager for handling a list of files on input. loads them asynchronously to be ready when called on. 
'''

# python imports
import torch
import numpy as np
import h5py
import argparse
from utils.reweight import reweight1d, reweight2d 

# Load data in agreed upon format
def loadDataFromH5(
        inFile, 
        eventSelection="", 
        loadWeights=False, 
        noLabels=False, 
        truthSB=True, 
        teacher=None, 
        split=[1,0,0],
        minCparam=0,
        minNjetsAbovePtCut=0,
        minNjets=0,
        reweight=0,
):
    with h5py.File(inFile, "r") as f:

        # energy
        e = np.array(f['source']['e'])
        e = np.log(e)
        e[e==-np.inf] = 0
        # pt
        pt = np.array(f['source']['pt'])
        pt = np.log(pt)
        pt[pt==-np.inf] = 0
        # phi
        phi = np.array(f['source']['phi'])
        # normalize eta
        eta = np.array(f['source']['eta'])
        # eta = (eta - np.mean(eta))/np.std(eta)
        # stack
        X = np.stack([pt,eta,np.cos(phi),np.sin(phi),e],-1)
        X = torch.Tensor(X)

        # load weights
        if loadWeights and 'normweight' in f['EventVars'].keys():
            W = np.array(f['EventVars/normweight'])
        else:
            W = None

        if noLabels and loadWeights:
            return X, torch.Tensor(W.reshape(-1,1)), None
        if noLabels:
            return X, None, None

        Y = np.array(f['EventVars/target_jet_assignments'])

        # load dsid
        dsid = np.array(f['EventVars/dsid'])
        
        # event selection
        event_selection = np.ones(Y.shape[0]).astype(bool)
        total_num_events = event_selection.sum()
        print("Total number of events: ", total_num_events)
        if "a" in eventSelection:
            # unlabeled_mask = np.array(f["g1"]["mask"]) * np.array(f["g2"]["mask"])
            unlabeled_mask = ((Y==0).sum(1) == 3) * ((Y==1).sum(1) == 3)
            # if teacher then update the unlabeled mask
            if teacher:
                with h5py.File(teacher,"r") as t:
                    # load predicted idx
                    pidx = np.array(t['trees_SRRPV_/pred_jet_assignments_set'])
                    # unlabled events that would not have been allowed in but have high prediction confidence are trained on
                    teacher_mask = (((pidx==0).sum(1) == 3) * ((pidx==1).sum(1) == 3))
                    teacher_idx = np.where(event_selection * (~unlabeled_mask) * teacher_mask)[0] # store which indices are new
                    Y[teacher_idx] = pidx[teacher_idx] # update Y
                    before = (event_selection*unlabeled_mask).sum() # store number of events without teacher
                    event_selection *= np.logical_or(unlabeled_mask, teacher_mask) # update event selection
                    print("Before, After, Delta teacher: ", before.sum(), event_selection.sum(), event_selection.sum()-before.sum())
            else:
                event_selection *= unlabeled_mask # hide unlabeled events
            print("Fully labeled: ", event_selection.sum())
        if "b" in eventSelection:
            pt = np.array(f['source']['pt'])
            C = np.array(f['EventVars/jet_Cparam'])
            event_selection *= (C>=minCparam)
            print("C-param cut: ", event_selection.sum())
            event_selection *= ((pt>=minNjetsAbovePtCut).sum(1) >= minNjets)
            print("NJet above pt cut: ", event_selection.sum())

        total_num_events = event_selection.sum()
        print("Final number of events: ", total_num_events)

        # append weights to Y
        if W is not None:
            Y = np.concatenate([Y,W.reshape(-1,1)],1)
        
        # make weights
        w = np.ones(event_selection.shape)
        if reweight != 0:
            target_ptetaphim = np.array(f['EventVars/target_ptetaphim'])
            target_mavg = target_ptetaphim[:,:,-1][:,:2].mean(1)
            ht = np.array(f['source/pt']).sum(1)
            dsid = np.array(f['EventVars/dsid'])
            smask = (dsid>=504509) * (dsid<=504552) * event_selection
            sx,sy,sv = target_mavg[smask], ht[smask], None #w[smask]
            bmask = (dsid>=364701) * (dsid<=364712) * event_selection
            bx,by,bv = target_mavg[bmask], ht[bmask], None #w[bmask]
            bins = [np.linspace(0,5000,1+50*2),np.linspace(1000,15000,1+150*2)]
            if reweight == 1:
                print("Doing 1D reweighting")
                sw, bw = reweight1d(sx,bx,bins[0],sv,bv,True)
                w[smask], w[bmask] = sw, bw
            elif reweight == 2:
                print("Doing 2D reweighting")
                sw, bw = reweight2d(sx,sy,bx,by,bins,sv,bv,True)
                w[smask], w[bmask] = sw, bw
            print("Signal weights min, max, mean, std: ", np.min(w[smask]),np.max(w[smask]),np.mean(w[smask]),np.std(w[smask]))
            print("Background weights min, max, mean, std: ", np.min(w[bmask]),np.max(w[bmask]),np.mean(w[bmask]),np.std(w[bmask]))
        Y = np.concatenate([Y,w.reshape(-1,1)],1)

        # append 1 (signal) or 0 (background)
        dsid = dsid.reshape(-1,1)
        if truthSB:
            dsid = ~((dsid>=364701) * (dsid<=364712))
        Y = np.concatenate([Y,dsid],1)

        # to tensor Y
        Y = torch.Tensor(Y)

        # default split will leave event_selection unchanged
        event_selection = train_val_test_split(event_selection=event_selection, total_num_events = total_num_events, train=split[0], val=split[1], test=split[2])            
        
    if len(np.where(X!=X)[0]) > 0 or len(np.where(Y!=Y)[0]) > 0 or len(np.where(W!=W)[0]) > 0:
        print("Checking for nan's in X: ", np.where(X!=X)[0])
        print("Checking for nan's in Y: ", np.where(Y!=Y)[0])
        print("Checking for nan's in W: ", np.where(W!=W)[0])
        
    return X, Y, event_selection

# use scheme fail event selection = 0, train set = 1, validation set = 2, test set = 3
def train_val_test_split(event_selection, total_num_events, train, val, test):
    splits = np.array([train,val,test]) # train, validation, test
    if np.round(splits.sum(),2) != 1:
        print(f"WARNING train, val, test splits don't sum to 1, they sum to {splits.sum()}!")
    splits *= total_num_events # convert to number of events
    print(total_num_events, splits, event_selection.sum(), event_selection.shape)
    splits = np.round(splits).astype(int) # at most off by 1 and that one event will just be left labeled as 0 or 1
    event_selection = event_selection.astype(int)
    # test set
    if (event_selection == 0).sum() > splits[2]:
        # if more unlabeled than test set size randomly choose some
        idx = np.random.choice(np.where(event_selection == 0)[0], splits[2], replace=False)
        event_selection[idx] = 3
    else:
        # if less unlabeled than desired test set size, then subtract and randomly pick more
        splits[2] -= (event_selection == 0).sum()
        event_selection[event_selection==0] = 3
        idx = np.random.choice(np.where(event_selection == 1)[0], splits[2], replace=False)
        event_selection[idx] = 3
    # validation set
    idx = np.random.choice(np.where(event_selection == 1)[0], splits[1], replace=False)
    event_selection[idx] = 2
    # count up
    # counts = np.array([[(event_selection==i).sum(),(event_selection==i).sum()/event_selection.shape[0]] for i in range(4)])
    # print(counts)
    return event_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",  "--inFile", default=None, help="Input file")
    parser.add_argument("--teacher", help="Teacher file", default=None)
    parser.add_argument("--event_selection", help="Do event selection", default="")
    parser.add_argument('--minNjetsAbovePtCut', default=100, type=int, help="Additional cut based on NJets with pt>=minNjetsAbovePtCut) >= minNjets. Typically 5 jets with pT>=100 GeV.")
    parser.add_argument('--minNjets', default=6, type=int, help="Minimum number of leading jets retained in h5 files. This counts the number of jets above minNjetsAbovePtCut.")
    parser.add_argument('--minCparam', default=0, type=float, help="Minimum value of the Cparameter")
    parser.add_argument('--reweight', default=0, type=int, help="Reweight 0(none), 1(1D), 2(2D)")
    ops = parser.parse_args()

    X, Y, idx = loadDataFromH5(
        ops.inFile, 
        eventSelection=ops.event_selection, 
        loadWeights=False,
        noLabels=False, 
        truthSB=True, 
        teacher=ops.teacher, 
        split=[0.7,0.2,0.1], 
        minCparam=ops.minCparam,
        minNjetsAbovePtCut=ops.minNjetsAbovePtCut,
        minNjets=ops.minNjets,
        reweight=ops.reweight
    )
    print(X.shape, Y.shape)
    #print([X[idx==i].shape for i in range(4)])
    #print(torch.where(torch.sum(X==0,(1,2)) == X.shape[1]*X.shape[2]))
