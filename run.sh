python3 train.py -i  inputs/Bkg.sampled_500k.h5 -o experiments/minimal/ -s 1000 -c config_files/minimal_config.json --sigFiles inputs/GG_qqq_1100.sampled_10k.h5 inputs/GG_qqq_1500.sampled_10k.h5
python3 evaluate.py -c config_files/minimal_config.json -i  inputs/Bkg.sampled_200k.h5 --noTruthLabels -o evaluate/ -w experiments/minimal/training_2023.06.16.15.06.07/finalWeights.ckpt --gpu
python3 tune.py -i   /home2/jmontejo/unsupervised-search/inputs/Bkg.sampled_500k.h5  --sigFiles /home2/jmontejo/unsupervised-search/inputs/GG_qqq_1100.sampled_10k.h5 /home2/jmontejo/unsupervised-search/inputs/GG_qqq_1500.sampled_10k.h5 -o experiments/scans -s 5000 --num_samples 128 -c config_files/minimal_config.json
python3 evaluate_scans_mp.py --bkg inputs/Bkg.sampled_20k.h5 --sig inputs/GG_qqq_*sampled_1k.h5  --gpu -w trained_model*/training*/cp*ckpt
python3 inspect_scans.py --infiles ` find . -name '*ckpt_summary.json'` --tight
for i in `cat inputs/chosen.txt`; do python3 evaluate.py -c trained_model_log/training_2023.06.27.21.48.22/lightning_logs/version_0/hparams.yaml -i  $i  -o evaluate_trained_log -w trained_model_log/training_2023.06.27.21.48.22/cp-epoch=0000-step=88.ckpt --gpu; done
