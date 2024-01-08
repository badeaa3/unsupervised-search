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
import yaml
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
    x = loadDataFromH5(config["inFileName"], ops.normWeights, ops.labels)
    if ops.labels:
        x, l = x
    elif ops.normWeights:
        x, w = x
    mask = (x[:,:,0] == 0)
    
    # evaluate
    outData = {}
    with torch.no_grad():

        # make predictions
        _loss, _prez, _z, _xloss, _candidates_p4, _jet_choice = [], [], [], [], [], []
        niters = int(np.ceil(x.shape[0]/ops.batch_size))
        for i in tqdm(range(niters)):
            start, end = i*ops.batch_size, (i+1)*ops.batch_size
            # be careful about the memory transfers to not use all gpu memory
            temp = x[start:end].to(config["device"])
            loss, mu_logvar, xloss, _, candidates_p4, jet_choice, masses = model(temp)
            masses, a,b,z0,z1,e,f = masses
            _prez.append(mu_logvar)
            z = torch.stack([z0,z1],dim=1)
            _z.append(z)
            _loss.append(loss)
            _xloss.append(xloss)
            _candidates_p4.append(candidates_p4)
            _jet_choice.append(jet_choice)

        # concat
        prez = torch.concat(_prez).cpu()
        z = torch.concat(_z).cpu()
        loss = torch.concat(_loss).cpu()
        xloss = torch.concat(_xloss).cpu()
        jet_choice = torch.concat(_jet_choice).cpu()
        candidates_p4 = torch.concat(_candidates_p4).cpu()
        
        # convert x
        x = x_to_p4(x)
        # apply mask to x
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)
        pmom_max, pidx_max = get_mass_max(x, jet_choice)

        # make output
        outData = {
            "jet_p4": x, # raw jets
            "prez": prez[...,-1], # MSE reco z
            "z": z, # MSE reco z
            "loss": loss[...,-1], # MSE reco loss
            "loss_crossed": xloss[...,-1], # MSE crossed reco loss
            "pred": jet_choice, # soft jet scores
            "pred_jet_assignments_max" : pidx_max, # interpreted prediction to jet assignments with max per jet
            "pred_ptetaphim_max" : pmom_max, # predicted 4-mom (pt,eta,phi,m)
        }
        if ops.labels:
            outData['labels'] = l
        if ops.normWeights:
            outData['normweight'] = w
        
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
    parser.add_argument("--labels",action="store_true", help="Store also sum of labels")
    parser.add_argument("-b", "--batch_size", help="Batch size", default=10**5, type=int)
    parser.add_argument('--event_selection', default="", help="Enable event selection in batcher.")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    parser.add_argument('--gpu', action="store_true", help="Run evaluation on gpu.")
    parser.add_argument('--noMassLoss', action="store_true", help="Remove mass from loss")
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
        if ops.config_file.endswith("yaml"):
            model_config = {
                       "batcher": {
                         "minCparam": 0,
                         "minNjetsAbovePtCut": 0,
                         "minNjets": 0,
                         "split": [
                           0.9,
                           0.1,
                           0.0
                         ],
                         "reweight": 0,
                         "eventSelection": "",
                         "teacher": False
                       },
                       "trainer": {
                         "precision": 32,
                         "gradient_clip_val": 0.1
                       },
                       "batch_size": 2048
                     }
            model_config = {}
            model_config["model"] = yaml.load(fp, Loader=yaml.Loader)
            #if 'lightning_logs' in ops.config_file:
            #    model_config['model']['encoder_config']['do_gumbel'] = True
            #    model_config['model']['encoder_config']['mass_scale'] = 100
        else:
            model_config = json.load(fp)
    if ops.noMassLoss:
            model_config["model"]["encoder_config"]["remove_mass_from_loss"] = True

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
