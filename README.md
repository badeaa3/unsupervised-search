# unsupervised-search
Unsupervised Machine Learning Based Approach to a Model Agnostic Search

## How to run
```
python3 train.py -i /eos/atlas/unpledged/group-tokyo/users/tsano/dataset/rpv_multijets/H5s/h5forML/Bkg.h5 -o experiments/0/ -s 100 -c config_files/minimal_config.json 
python3 evaluate.py -c config_files/minimal_config.json -i /eos/atlas/unpledged/group-tokyo/users/tsano/dataset/rpv_multijets/H5s/h5forML/GG_qqq_1500.h5 --noTruthLabels -o evaluate/ -w experiments/0/training_2023.06.16.21.57.52/finalWeights.ckpt --gpu --doOverwrite
```
