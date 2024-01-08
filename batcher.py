'''
batch manager for handling a list of files on input. loads them asynchronously to be ready when called on. 
'''

# python imports
import torch
import numpy as np
import h5py
import argparse

# Load data in agreed upon format
def loadDataFromH5(
        inFile, 
        normWeights=False,
        labels=False
):
    with h5py.File(inFile, "r") as f:

        # energy
        e = np.array(f['source']['e'])
        e = np.nan_to_num(e)
        e = np.log(e)
        e[e==-np.inf] = 0
        # pt
        pt = np.array(f['source']['pt'])/1000. #takane input has wrong units for pt and mass
        pt = np.nan_to_num(pt)
        pt = np.log(pt)
        pt[pt==-np.inf] = 0
        # phi
        phi = np.array(f['source']['phi'])
        # eta
        eta = np.array(f['source']['eta'])

        #mask = pt<50
        #e[mask], pt[mask], phi[mask], eta[mask] = 0,0,0,0
        #selection cuts
        njet = np.sum(e>0,-1)
        cuts = njet >= 6
        # stack
        X = np.stack([pt,eta,np.cos(phi),np.sin(phi),e],-1)
        X = X[cuts]
        X = torch.Tensor(X)

        if labels:
            l = np.array(f['source']['label'])
            l = l[cuts]
            X = (X,l)
        elif normWeights:
            w = np.array(f['EventVars']['normweight'])
            w = w[cuts]
            X = (X,w)
        
    if len(np.where(X!=X)[0]) > 0:
        print("Checking for nan's in X: ", np.where(X!=X)[0])
        
    return X

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",  "--inFile", default=None, help="Input file")
    ops = parser.parse_args()

    X = loadDataFromH5(
        inFile = ops.inFile,
    )
    print(X.shape)
    print(X[0])
