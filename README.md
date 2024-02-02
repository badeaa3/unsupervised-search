# Paper details
Title: A data-driven and model-agnostic approach to solving combinatorial assignment problems in searches for new physics \
Authors: Anthony Badea, Javier Montejo Berlingen \
arxiv: https://arxiv.org/abs/2309.05728 \ 
PRD: https://journals.aps.org/search \

# How to reproduce results
```
python3 train.py -i  inputs/Bkg.sampled_500k.h5 -o experiments/minimal/ -s 1000 -c config_files/minimal_config.json --sigFiles inputs/GG_qqq_1100.sampled_10k.h5 inputs/GG_qqq_1500.sampled_10k.h5 --device gpu

python3 evaluate.py -c config_files/minimal_config.json -i  inputs/GG_qqq_1500.h5 -o evaluate/ -w experiments/minimal/training_2024.02.02.06.19.38/finalWeights.ckpt

python3 evaluate.py -c config_files/minimal_config.json -i  inputs/Bkg.sampled_200k.h5 -o evaluate/ -w experiments/minimal/training_2024.02.02.06.19.38/finalWeights.ckpt

# then plot
python
> import numpy as np;import h5py;import matplotlib.pyplot as plt
> s = h5py.File("evaluate/GG_qqq_1500_transformer_classifier.h5","r")
> b = h5py.File("evaluate/Bkg_transformer_classifier.h5","r")
> smavg = np.array(s["pred_ptetaphim_max"][:,:,-1].mean(-1))
> bmavg = np.array(b["pred_ptetaphim_max"][:,:,-1].mean(-1))
> bins=np.linspace(0,3500,36);density=True;plt.hist(bmavg, density=density, bins=bins, histtype="step", color="black", label="bkg");plt.hist(smavg, density=density, bins=bins, histtype="step", color="blue", label="GG_qqq_1500");plt.legend();plt.show()
```
