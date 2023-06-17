# unsupervised-search
Unsupervised Machine Learning Based Approach to a Model Agnostic Search

## How to run
```
./train.py -i /eos/atlas/unpledged/group-tokyo/users/tsano/dataset/rpv_multijets/H5s/h5forML/Bkg.h5
python train.py -i /eos/atlas/unpledged/group-tokyo/users/tsano/dataset/rpv_multijets/H5s/h5forML/Bkg.h5 -o experiments/0/ -s 100
```
## To discuss

* Removed the 3 score features from the AE, those should not be predicted
* Right now I feed mass as additional input to the AE, and then also into the latent space, but don't attempt to predict it
* We are injecting the AE within one of the big blocks, after the candidate attention
   * I think this allows candidates to learn their "identity" and trick the AE (see below)
   * Should probably put the AE at the end of a big block, simply building candidates from the last jet score
* There seem to be a few bad situations in the training:
   * it seems the AE is able to recognize the random combination, then predicts arbitrarily large values to maximize the distance and the loss diverges
   * the embedding sometimes maps all jets to the same value, so that there is no difference between jets. In this way regardless of the combination choice the candidates are identical
   * also when the embedding works ok, often the predictions are 50% g1/g2, so that the candidates are identical (due to soft gumbel) 
   * I don't get hard gumbel to run (nans), same if the annealing brings tau < 0.2 or so. Check gradient clipping
