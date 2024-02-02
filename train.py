#! /usr/bin/env python

'''
Author: Anthony Badea
Date: November 16, 2022
'''

# python packages
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import h5py
import datetime
import json
import yaml

# custom code
from batcher import loadDataFromH5
from model import StepLightning

if __name__ == "__main__":

    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./config_files/default_config.json")
    parser.add_argument("-i", "--inFile", help="Input training file.", default=None, required=True)
    parser.add_argument("--sigFiles", nargs="+", help="Optional signal files to use for validation", default=[])
    parser.add_argument("-o", "--outDir", help="File name of the output directory", default="./checkpoints")
    parser.add_argument("-e", "--max_epochs", help="Max number of epochs to train on", default=None, type=int)
    parser.add_argument("-s", "--max_steps", help="Max number of steps to train on", default=-1, type=int)
    parser.add_argument("-d", "--device", help="Device to use.", default=None)
    parser.add_argument("-w", "--weights", help="Initial weights.", default=None)
    parser.add_argument("--update-encoder", help="Dictionary to update config['model']['encoder_config'] from CLI", default={})
    parser.add_argument("--update-loss",    help="Dictionary to update config['model']['loss_config'] from CLI", default={})
    ops = parser.parse_args()

    # load configuration
    print(f"Using configuration file: {ops.config_file}")
    with open(ops.config_file, 'r') as fp:
        if ops.config_file.endswith("yaml"):
            from batcher_defaults import config
            config["model"] = yaml.load(fp, Loader=yaml.Loader)
            if 'lightning_logs' in ops.config_file:
                config['model']['encoder_config']['do_gumbel'] = False
                config['model']['encoder_config']['mass_scale'] = 10
        else:
            config = json.load(fp)
    if ops.update_encoder:
        config['model']['encoder_config'].update(eval(ops.update_encoder))
    if ops.update_loss:
        config['model']['loss_config'].update(eval(ops.update_loss))
    print(config)

    config["model"]["weights"] = ops.weights

    # decide on device
    device = ops.device
    if not device:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "gpu")

    # load and split
    X = loadDataFromH5(ops.inFile)
    Xsig = [loadDataFromH5(sig) for sig in ops.sigFiles]
    X_train, X_val = train_test_split(X, test_size = 0.1)
    print(f"X_train {X_train.shape}, X_val {X_val.shape}")
    train_dataloader = DataLoader(X_train, shuffle=True, num_workers=4, pin_memory=pin_memory, batch_size=config["batch_size"]) # if multiple inputs beside just X then use DataLoader(TensorDataset(X, ...), ...)
    val_dataloader = DataLoader(X_val, shuffle=False, num_workers=2, pin_memory=pin_memory, batch_size=config["batch_size"])
    valsig_dataloaders = [DataLoader(sig, shuffle=False, num_workers=2, pin_memory=pin_memory, batch_size=config["batch_size"]//2) for sig in Xsig]
    
    # make checkpoint dir
    checkpoint_dir = os.path.join(ops.outDir, f'training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}')
    print(f"Saving checkpoints to: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create model
    model = StepLightning(device="cuda" if device == "gpu" else "cpu", **config["model"])
    #model = torch.compile(model) # ready for pytorch 2.0 once it's more stable

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_loss/dataloader_idx_0", dirpath=checkpoint_dir, filename='cp-val_loss-{val_loss:.2f}-{std_1:.2f}-{std_2:.2f}-{epoch:04d}-{step}', save_top_k=1), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        ModelCheckpoint(monitor="std_1", dirpath=checkpoint_dir, filename='cp-std1-{val_loss:.2f}-{std_1:.2f}-{std_2:.2f}-{epoch:04d}-{step}', save_top_k=1), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        ModelCheckpoint(monitor="std_2", dirpath=checkpoint_dir, filename='cp-std2-{val_loss:.2f}-{std_1:.2f}-{std_2:.2f}-{epoch:04d}-{step}', save_top_k=1), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        EarlyStopping(monitor="val_loss/dataloader_idx_0", patience=3),
    ]

    # torch lightning trainer
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=ops.max_epochs,
        max_steps=ops.max_steps,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        #detect_anomaly=True,
        **config["trainer"]
    )
    
    # fit
    trainer.fit(model, train_dataloader, val_dataloaders = [val_dataloader]+valsig_dataloaders)
    
    # save model
    trainer.save_checkpoint(os.path.join(checkpoint_dir,"finalWeights.ckpt"))
