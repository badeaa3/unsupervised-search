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
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
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
    parser.add_argument("--num_samples", help="Number of trails to run", default=2, type=int)
    return parser.parse_args()

def train(config, init_config={}, inFile="", outDir="", max_steps = 100, device = "cpu", num_gpus = 0):
    
    # update init_config with config
    init_config["model"]["encoder_config"]["embed_dim"] = config["embed_dim"]
    init_config["model"]["encoder_config"]["attn_blocks_n"] = config["attn_blocks_n"]
    init_config["lr"] = config["lr"]
    init_config["batch_size"] = config["batch_size"]

    # load data and split
    X, Y, idx = loadDataFromH5(inFile, loadWeights=False, noLabels=False, truthSB=True, **init_config["batcher"])
    X_train, Y_train = X[idx==1], Y[idx==1]
    X_val, Y_val = X[idx==2], Y[idx==2]
    
    # make data loaders
    num_workers = 4
    pin_memory = (device == "gpu")
    train_dataloader = DataLoader(TensorDataset(X_train, Y_train), shuffle=True, num_workers=num_workers, pin_memory=pin_memory, batch_size=init_config["batch_size"])
    val_dataloader = DataLoader(TensorDataset(X_val, Y_val), shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=init_config["batch_size"])
    
    # make checkpoint dir
    checkpoint_dir = os.getcwd()

    # save event selection idx to file
    with h5py.File(os.path.join(checkpoint_dir,"event_selection.h5"), "w") as f:
        f.create_dataset("idx", data = idx)

    # create model
    model = StepLightning(**init_config["model"])

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="train_loss", dirpath=checkpoint_dir, filename='cp-{epoch:04d}-{step}', every_n_train_steps = 1, save_top_k=20), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        TuneReportCallback({ "val_loss" : "val_loss", "val_acc": "val_acc" }, on="validation_end")
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
    trainer.fit(model, train_dataloader, val_dataloader)
    
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

    # configure search space
    config = {
        "embed_dim" : tune.choice([32, 64, 128]),
        "attn_blocks_n" : tune.choice([4, 6, 8, 10]),
        "lr" : 1e-3, #tune.loguniform(1e-4, 1e-1), 
        "batch_size" : 2048, #tune.choice([1024, 2048, 4096])
    }

    # make scheduler
    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([512, 1024, 2048, 4096])
        }
    )

    # change the CLI output
    reporter = CLIReporter(
        parameter_columns=["embed_dim", "attn_blocks_n", "lr", "batch_size"],
        metric_columns=["val_loss", "val_acc", "training_iteration"]
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
            metric="val_loss",
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

