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
    parser.add_argument("--doSignalsOnly",action="store_true", help="Check only resolution on GG_qqq samples")
    parser.add_argument("--do1500",action="store_true", help="Check only resolution on GG_qqq_1500 sample")
    parser.add_argument("--m0",action="store_true", help="Impose m0 is used sensibly in T3 trainings")
    parser.add_argument("--tight",action="store_true", help="Apply a tight skim of the results: mavg>1.28 and loss>0.8")
    parser.add_argument("--corr",action="store_true", help="Require at least 50% correlation")
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
      try:
          k,v = t.split("=")
          ret[k] = eval(v)
      except ValueError: pass
    return ret
 
if __name__ == "__main__":

    # user options
    args = options()
    metametrics = {}
    metricslist = ['mavg','loss','mavg_i0','mavg_i1','mavg_i2','mavg_i3']
    metricslist = ['mavg','loss','mavg_i0','mavg_i1']
    metricslist = ['mavg','loss','corr_max_z_mavg']
    for infile in args.infiles:
        if infile.endswith("ckpt"):
             infile += "_summary.json"
        if not os.path.exists(infile): continue
        paramfile = os.path.dirname(infile)+"/params.json"
        if os.path.exists(paramfile):
            pf = open(paramfile)
            params  = json.load(pf)
        else: params = {}
        with open(infile) as f:
            metrics = json.load(f)
            if not all(m in metrics for m in metricslist): continue
            metrics['aggregate'] = get_aggregate(metrics) #overwrite aggregate
            metametrics[infile] = {'aggregate':metrics['aggregate']}
            for var in metricslist+['m0']:
                if var=='aggregate': continue
                if var == 'corr_max_z_mavg':
                    metametrics[infile][var] = metrics[var]['bkg']
                elif var == 'loss':
                    metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1100","GG_qqq_1500")])
                #    metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1100",)])
                elif args.do1500 and var =='mavg':
                    metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1500",)])
                elif args.doSignalsOnly and var =='mavg':
                    metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1100","GG_qqq_1500","GG_qqq_1900","GG_qqq_2300")])
                else:    
                    metametrics[infile][var] = metrics[var]['avg']
                #if var!='mavg':
                #    metametrics[infile][var] = metrics[var]['avg']
                #else:
                #    metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1100","GG_qqq_1500","GG_qqq_1900","GG_qqq_2300")])
                #    #metametrics[infile][var] = np.mean([metrics[var][x] for x in ("GG_qqq_1100",)])
                if var=='mdiff':
                     if metametrics[infile][var] == 0:  metametrics[infile][var] = 999999
                     metametrics[infile][var] = -metametrics[infile][var]
            metametrics[infile].update(params)
            #if isnan(metrics['outdiff']['avg']) and metrics['aggregate']!=0 : print(infile)
        if os.path.exists(paramfile):
            pf.close()

    #metametrics = {k:v for k, v in sorted(metametrics.items(), key=lambda item: item[1][var]) if not isnan(v[var]) and v['mavg'] > 1 and v['loss'] > 0.6 and (not args.m0 or v['m0']!=0.812867520143782) and (not args.tight or (v['mavg']>3.5 and v['loss']>0.75))}
    metametrics = {k:v for k, v in sorted(metametrics.items(), key=lambda item: item[1][var]) if not isnan(v[var]) and (not args.corr or v['corr_max_z_mavg']>0.5) and (not args.m0 or v['m0']!=0.812867520143782) and (not args.tight or (v['mavg']>3.75 and v['loss']>0.85))}
    best = set()
    for var in metricslist+['aggregate']:
        print(var)
        skimmed = [(k,v['mavg'],v['loss'],v['corr_max_z_mavg'],v['aggregate']) for k, v in sorted(metametrics.items(), key=lambda item: item[1][var])]
        print(len(skimmed))
        top = 10
        for t in skimmed[-top:]:
            print(var, "->", t)
            best.add(t[0])


    #print("Best set")
    #for b in best:
    #    print(b, metametrics[b])
    if args.cp:
        for b in best:
            print(f"cp {b.replace('json','h5')} /eos/home-j/jmontejo/unsupervised_outputs")

    with open('metametrics.json','w') as outfile:
        print(json.dump(metametrics, outfile))
