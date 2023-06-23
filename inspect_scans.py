#! /usr/bin/env python

'''
Author: Anthony Badea
'''

# python packages
import argparse
import numpy as np
import os, sys
import h5py
import json
import yaml
import glob
from math import isnan
from matplotlib import pyplot as plt

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--infiles", nargs="+", help="Json file(s) to inspect", default=None, required=True)
    parser.add_argument("--m0",action="store_true", help="Impose m0 is used sensibly in T3 trainings")
    parser.add_argument("--tight",action="store_true", help="Apply a tight skim of the results: mavg>1.28 and loss>0.8")
    parser.add_argument("--cp",action="store_true", help="print out the commands to copy to eos")
    return parser.parse_args()

def plot(infile):

    with h5py.File(infile, 'r') as f:
        for var in f['bkg'].keys():
            histos = {}
            bins = np.arange(len(f['bkg'][var]))
            for sample in f.keys():
                h = np.array(f[sample][var])
                plt.bar(bins, h/sum(h), alpha=0.3, label=sample)
            plt.xlabel(var)
            plt.legend()
            plt.savefig(infile.replace('.h5',"_"+var+".png"))
            plt.close()

def get_aggregate(metrics):
    metrics = {k:v['avg'] for k, v in metrics.items() if k!='aggregate'}
    if any(isnan(v) for v in metrics.values()):
        return 0
    separation =['loss',]
    agg = max([metrics[k] for k in separation])
    agg += metrics['mavg']/2
    #agg -= metrics['mdiff']/100
    return agg

def parse_name(infile):
    #experiments/scans/tune_2023.06.22.11.15.40/train_5dafc_00015_15_ae_dim=8,do_gumbel=True,energyT=True,mass_scale=40,out_dim=3,scale_ISR_loss=0.0500_2023-06-22_11-28-12
    infile = infile.split("/")[-2]
    tokens = infile.split(",")
    if len(tokens)<3: return {}
    tokens[0] = "ae_"+tokens[0].split("_")[-1]
    tokens[-1] = "scale_ISR_"+tokens[-1].split("_")[2]
    ret = {}
    for t in tokens:
      k,v = t.split("=")
      ret[k] = eval(v)
    return ret
 
if __name__ == "__main__":

    # user options
    args = options()
    metametrics = {}
    metricslist = ['mavg','loss']
    for infile in args.infiles:
        with open(infile) as f:
            metrics = json.load(f)
            metrics['aggregate'] = get_aggregate(metrics) #overwrite aggregate
            metametrics[infile] = {'aggregate':metrics['aggregate']}
            for var in metricslist+['m0']:
                if var=='aggregate': continue
                metametrics[infile][var] = metrics[var]['avg']
                #if var!='mavg':
                #    metametrics[infile][var] = metrics[var]['avg']
                #else:
                #    metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1100","GG_qqq_1500","GG_qqq_1900","GG_qqq_2300")])
                if var=='mdiff':
                     if metametrics[infile][var] == 0:  metametrics[infile][var] = 999999
                     metametrics[infile][var] = -metametrics[infile][var]
            metametrics[infile].update(parse_name(infile))
            #if isnan(metrics['outdiff']['avg']) and metrics['aggregate']!=0 : print(infile)

    metametrics = {k:v for k, v in sorted(metametrics.items(), key=lambda item: item[1][var]) if not isnan(v[var]) and v['mavg'] > 1 and v['loss'] > 0.7 and (not args.m0 or v['m0']!=0.8) and (not args.tight or (v['mavg']>3 and v['loss']>0.85))}
    best = set()
    for var in metricslist+['aggregate']:
        print(var)
        skimmed = [(k,v[var],v['aggregate']) for k, v in sorted(metametrics.items(), key=lambda item: item[1][var])]
        print(len(skimmed))
        top = 12
        for t in skimmed[-top:]:
            print(var, "->", t)
            best.add(t[0])


    print("Best set")
    for b in best:
        print(b, metametrics[b])
    if args.cp:
        for b in best:
            print(f"cp {b.replace('json','h5')} /eos/home-j/jmontejo/unsupervised_outputs")

    with open('metametrics.json','w') as outfile:
        print(json.dump(metametrics, outfile))
