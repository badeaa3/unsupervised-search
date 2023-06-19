#! /usr/bin/env python

'''
Author: Anthony Badea
'''

# python packages
import torch
import argparse
import numpy as np
import os
import h5py
import json
import glob
from tqdm import tqdm 
from model_blocks import x_to_p4

# multiprocessing
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# custom code
from batcher import loadDataFromH5
from model import StepLightning
from utils.get_mass import get_mass_max, get_mass_set

def evaluate(config):
    ''' perform the full model evaluation '''
    
    ops = options()
    config["model"]["weights"] = ops.weights

    print(f"evaluating on {config['inFileName']}")

    # load model
    model = StepLightning(**config["model"])
    model.to(config["device"])
    model.eval()
    model.Encoder.eval()

    # load data
    x = loadDataFromH5(config["inFileName"], ops.normWeights)
    if ops.normWeights:
        x, w = x
    mask = (x[:,:,0] == 0)
    
    # evaluate
    outData = {}
    with torch.no_grad():

        # make predictions
        p, ae = [], []
        niters = int(np.ceil(x.shape[0]/ops.batch_size))
        for i in tqdm(range(niters)):
            start, end = i*ops.batch_size, (i+1)*ops.batch_size
            # be careful about the memory transfers to not use all gpu memory
            temp = x[start:end].to(config["device"])
            ae_out, jet_choice, scores, interm_masses = model(temp)
            c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4 = ae_out
            c1, c2, c1_out, c2_out = c1.cpu(), c2.cpu(), c1_out.cpu(), c2_out.cpu()
            jet_choice = jet_choice.cpu()
            ae.append(torch.stack([c1, c2, c1_out, c2_out],-1))
            p.append(jet_choice)

        # concat
        p = torch.concat(p)
        ae = torch.concat(ae)
        c1, c2, c1_out, c2_out = [ae[:,i] for i in range(4)]
        mse_loss = torch.mean((c1_out-c1)**2 + (c2_out-c2)**2,-1)
        mse_crossed_loss = torch.mean((c1_out-c2)**2 + (c2_out-c1)**2,-1)
        
        # convert x
        x = x_to_p4(x)
        # apply mask to x
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)
        pmom_max, pidx_max = get_mass_max(x, p)
                
        # make output
        outData = {
            "loss": mse_loss.numpy(), # raw prediction
            "loss_crossed": mse_crossed_loss.numpy(), # raw prediction
            "pred": p.numpy(), # raw prediction
            "jet_p4": x.numpy(), # raw jets
            "pred_jet_assignments_max" : pidx_max.numpy(), # interpreted prediction to jet assignments with max per jet
            "pred_ptetaphim_max" : pmom_max.cpu().numpy(), # predicted 4-mom (pt,eta,phi,m)
        }
        if ops.normWeights:
            outData['normweight'] = w
        
        # if truth labels then do y
        if not ops.noTruthLabels:
            y = y.cpu()
            # convert y to one-hot and get mass
            ymass = y[:,:-2].numpy().astype(int)
            n_values = np.max(ymass) + 1
            ymass = torch.Tensor(np.eye(n_values)[ymass])
            ymom, yidx = get_mass_max(x, ymass)
            outData["target_jet_assignments"] = y[:,:-2].cpu().numpy() # target jet assignments
            outData["target_ptetaphim"] = ymom.cpu().numpy() # target 4-mom

    # save final file
    print(f"Saving to {config['outFileName']}")
    with h5py.File(config['outFileName'], 'w') as hf:
        for key, val in outData.items():
            print(f"{key} {val.shape}")
            hf.create_dataset(key, data=val)
    print("Done!")

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./config_files/default_config.json")
    parser.add_argument("-i",  "--inFile", help="Data file to evaluate on.", default=None, required=True)
    parser.add_argument("-o",  "--outDir", help="Directory to save evaluation output to.", default="./")
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-w",  "--weights", help="Pretrained weights to evaluate with.", default=None, required=True)
    parser.add_argument("--normWeights",action="store_true", help="Store also normalization weights")
    parser.add_argument("-b", "--batch_size", help="Batch size", default=10**5, type=int)
    parser.add_argument('--event_selection', default="", help="Enable event selection in batcher.")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    parser.add_argument('--noTruthLabels', action="store_true", help="Option to tell data loader that the file does not contain truth labels")
    parser.add_argument('--gpu', action="store_true", help="Run evaluation on gpu.")
    return parser.parse_args()
 
if __name__ == "__main__":

    # user options
    ops = options()

    # load input data
    data = ops.inFile
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        data = [data]
    elif os.path.isfile(data) and ".root" in os.path.basename(data):
        data = [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        data = sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        data = sorted(os.listdir(data))
    elif "*" in data:
        data = sorted(glob.glob(data))

    # make output dir
    if not os.path.isdir(ops.outDir):
        os.makedirs(ops.outDir)

    # pick up model configurations
    print(f"Using configuration file: {ops.config_file}")
    with open(ops.config_file, 'r') as fp:
        model_config = json.load(fp)

    # understand device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if ops.gpu else "cpu"

    # create evaluation job dictionaries
    config  = []
    for inFileName in data:

        # make out file name and check if already exists
        outFileName = os.path.join(ops.outDir, os.path.basename(inFileName)).replace(".h5","_transformer_classifier.h5")
        if os.path.isfile(outFileName) and not ops.doOverwrite:
            print(f"File already exists not evaluating on: {outFileName}")
            continue

        # append configuration
        config.append({
            "inFileName" : inFileName,
            "outFileName" : outFileName,
            "device" : device,
            **model_config
        })

    # launch jobs
    if ops.ncpu == 1:
        for conf in config:
            evaluate(conf)
    else:
        results = mp.Pool(ops.ncpu).map(evaluate, config)
