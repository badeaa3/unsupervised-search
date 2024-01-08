#! /usr/bin/env python

'''
Author: Anthony Badea
Date: April 15, 2023
'''

# python packages
import os
import argparse
import h5py
import datetime
import json
import math
from sklearn.model_selection import train_test_split

# torch
import torch
from torch.utils.data import DataLoader, TensorDataset

# torch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# ray tune
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from ray.tune import CLIReporter

# custom code
from batcher import loadDataFromH5
from model import StepLightning

def options():
    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./config_files/default_config.json")
    parser.add_argument("-i", "--inFile", help="Input training file.", default=None, required=True)
    parser.add_argument("-o", "--outDir", help="File name of the output directory", default="./checkpoints")
    parser.add_argument("-e", "--max_epochs", help="Max number of epochs to train on", default=None, type=int)
    parser.add_argument("-s", "--max_steps", help="Max number of steps to train on", default=-1, type=int)
    parser.add_argument("-d", "--device", help="Device to use.", default=None)
    parser.add_argument("-w", "--weights", help="Initial weights.", default=None)
    parser.add_argument("--num_samples", help="Number of trails to run", default=2, type=int)
    parser.add_argument("--sigFiles", nargs="+", help="Optional signal files to use for validation", default=[])
    return parser.parse_args()

def train(config, init_config={}, inFile="", outDir="", max_steps = 100, device = "cpu", num_gpus = 0):
    
    # update init_config with config
    #init_config["model"]["encoder_config"]["ae_dim"] = config["ae_dim"]
    init_config["model"]["encoder_config"]["do_vae"] = config["do_vae"]
    init_config["model"]["encoder_config"]["out_dim"] = config["out_dim"]
    #init_config["model"]["encoder_config"]["do_gumbel"] = config["do_gumbel"]
    init_config["model"]["encoder_config"]["mass_scale"] = config["mass_scale"]
    #init_config["model"]["encoder_config"]["add_mass_feature"] = config["add_mass_feature"]
    #init_config["model"]["encoder_config"]["add_mass_latent"] = config["add_mass_latent"]
    init_config["model"]["encoder_config"]["sync_rand"] = config["sync_rand"]
    #init_config["model"]["encoder_config"]["over_jet_count"] = config["over_jet_count"]
    #init_config["model"]["encoder_config"]["random_mode"] = config["random_mode"]
    #init_config["model"]["encoder_config"]["remove_mass_from_loss"] = config["remove_mass_from_loss"]
    init_config["model"]["encoder_config"]["rand_cross_candidates"] = config["rand_cross_candidates"]
    init_config["model"]["L2"] = config["L2"]
    init_config["model"]["loss_config"]["scale_ISR_loss"] = config["scale_ISR_loss"]
    init_config["model"]["loss_config"]["scale_random_loss"] = config["scale_random_loss"]
    init_config["model"]["loss_config"]["scale_latent_loss"] = config["scale_latent_loss"]
    init_config["model"]["loss_config"]["scale_kld_loss"] = config["scale_kld_loss"]
    init_config["model"]["loss_config"]["scale_reco_loss"] = config["scale_reco_loss"]

    # load data and split
    X = loadDataFromH5(inFile)
    Xsig = [loadDataFromH5(sig) for sig in ops.sigFiles]
    X_train, X_val = train_test_split(X, test_size = 0.1)
    
    # make data loaders
    num_workers = 1
    pin_memory = (device == "gpu")
    train_dataloader = DataLoader(X_train, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, batch_size=init_config["batch_size"]//2)
    val_dataloader = DataLoader(X_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=init_config["batch_size"]//2)
    valsig_dataloaders = [DataLoader(sig, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=init_config["batch_size"]//2) for sig in Xsig]
    
    # make checkpoint dir
    checkpoint_dir = os.getcwd()

    # create model
    model = StepLightning(**init_config["model"])

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_loss/dataloader_idx_0", dirpath=checkpoint_dir, filename='cp-val_loss-{val_loss:.2f}-{std_1:.2f}-{std_2:.2f}-{epoch:04d}-{step}', save_top_k=1), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        ModelCheckpoint(monitor="std_1", mode = 'max', dirpath=checkpoint_dir, filename='cp-std1-{val_loss:.2f}-{std_1:.2f}-{std_2:.2f}-{epoch:04d}-{step}', save_top_k=1), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        ModelCheckpoint(monitor="std_2", mode = 'max', dirpath=checkpoint_dir, filename='cp-std2-{val_loss:.2f}-{std_1:.2f}-{std_2:.2f}-{epoch:04d}-{step}', save_top_k=1), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        EarlyStopping(monitor="val_loss/dataloader_idx_0", patience=3),
        TuneReportCallback({ "val_loss/dataloader_idx_0" : "val_loss/dataloader_idx_0"}, on="validation_end")
    ]

    # torch lightning trainer
    trainer = pl.Trainer(
        accelerator=device,
        devices=math.ceil(num_gpus),
        max_steps = max_steps,
        logger=TensorBoardLogger(save_dir=checkpoint_dir, name="", version="."),
        #log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        enable_progress_bar=False,
        **init_config["trainer"]
    )
    
    # fit
    trainer.fit(model, train_dataloader,  val_dataloaders = [val_dataloader]+valsig_dataloaders)
    
if __name__ == "__main__":

    # user options
    ops = options()

    # decide on device
    device = ops.device
    if not device:
        device = "gpu" if torch.cuda.is_available() else "cpu"

    # load configuration
    print(f"Using configuration file: {ops.config_file}")
    with open(ops.config_file, 'r') as fp:
        init_config = json.load(fp)
    init_config["model"]["weights"] = ops.weights

    # configure search space
    config = {
        #"do_gumbel" : tune.choice([False]),
        #"ae_dim"    : tune.choice([2]),
        "do_vae": tune.choice([True]),
        #"add_mass_feature": tune.choice([True]),
        #"add_mass_latent": tune.choice([False]),
        #"over_jet_count": tune.choice([True]),
        "out_dim"   : tune.choice([3,[2,3]]),
        "scale_ISR_loss" : tune.choice([0.1, 0.2, 0.5]),
        "scale_random_loss" : tune.choice([0.01, 0.001, 0.0001]),
        "scale_latent_loss" : tune.choice([0.05, 0.1]),
        "scale_kld_loss" : tune.choice([0.01, 0.1, 1]),
        "scale_reco_loss" : tune.choice([10, 100]),
        #"remove_mass_from_loss" : tune.choice([False]),
        "mass_scale" : tune.choice([10, 50, 100]),
        #"mass_scale" : tune.choice([0.3, 1, 3, 10]),
        "sync_rand": tune.choice([True,False]),
        #"random_mode": tune.choice(['reverse_both']),
        "rand_cross_candidates": tune.choice([True,False]),
        "L2": tune.choice([1e-1,1e-2,1e-3]),
    }

    # make scheduler
    scheduler = FIFOScheduler()

    # change the CLI output
    reporter = CLIReporter(
        parameter_columns = list(config.keys()),
        metric_columns=["val_loss/dataloader_idx_0", "training_iteration"]
    )

    # tune with parameters
    gpus_per_trial = 0.5
    train_fn_with_parameters = tune.with_parameters(train, init_config=init_config, inFile=ops.inFile, outDir=ops.outDir, max_steps=ops.max_steps, device=device, num_gpus=gpus_per_trial)

    # configure resources
    resources_per_trial = {"cpu": 4, "gpu": gpus_per_trial}

    # tune
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss/dataloader_idx_0",
            mode="min",
            scheduler=scheduler,
            num_samples=ops.num_samples,
        ),
        run_config=air.RunConfig(
            name=f"tune_{datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}",
            progress_reporter=reporter,
            local_dir=ops.outDir,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


