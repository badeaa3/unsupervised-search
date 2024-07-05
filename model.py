import torch
import pytorch_lightning as pl
import model_blocks 
from model_blocks import m2s_from_p4s
from utils.functions import distance_corr

import numpy as np
from matplotlib import pyplot as plt
import io
from PIL import Image
import torchvision

class StepLightning(pl.LightningModule):

    def __init__(self,
                 encoder_config = {},
                 loss_config = {},
                 lr = 1e-3,
                 update_learning_rate = True,
                 tau_annealing = True,
                 weights = None):
        super().__init__()

        self.Encoder = model_blocks.Encoder(**encoder_config)
        self.encoder_config = encoder_config
        self.loss_config = loss_config
        self.lr = lr
        self.update_learning_rate = update_learning_rate
        self.tau_annealing = tau_annealing
        self.T = encoder_config["out_dim"]

        # use the weights hyperparameters
        if weights: 
            ckpt = torch.load(weights,map_location=self.device)
            self.load_state_dict(ckpt["state_dict"])
        
        self.save_hyperparameters()

        # store recent performance in validation step
        self._recent_perf = {
            'm0': [],
            'm1': [],
            'loss': [],
        }

    def forward(self, x):

        # create covariant angular piece [eta, cos(phi), sin(phi)]
        w = torch.stack([x[:,:,1], x[:,:,2], x[:,:,3]],-1)

        # create padding mask with True where padded
        mask = (x[:,:,0] == 0).bool()
        loss, xloss, candidates_p4, jet_choice = self.Encoder(x, w, mask)
        return loss, xloss, candidates_p4, jet_choice
        
    def step(self, batch, batch_idx, version):
        
        # train update learning rate
        if version == "train" and self.update_learning_rate:
            self.learning_rate_scheduler()
        pg = self.trainer.optimizers[0].param_groups[0]
        self.log("lr", pg["lr"], prog_bar=True, on_step=True)

        if version == "train" and self.tau_annealing:
            self.encoder_config["gumbel_softmax_config"]["tau"] *= 1-1./self.trainer.max_steps #converges to 0.36*original tau
            self.log("tau", self.encoder_config["gumbel_softmax_config"]["tau"], prog_bar=True, on_step=True)

        # forward pass
        x = batch
        loss, xloss, candidates_p4, jet_choice = self(x)

        # compute loss
        loss = self.loss(loss, xloss, candidates_p4)

        # log the loss
        for key, val in loss.items():
            self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=True)
        
        # log the performance
        if version == "val":
            m2s = m2s_from_p4s(candidates_p4)
            self._recent_perf['m0'].append(m2s[:,0].sqrt().detach().cpu().numpy())
            self._recent_perf['m1'].append(m2s[:,1].sqrt().detach().cpu().numpy())
            self._recent_perf['loss'].append(loss["loss"].detach().cpu().numpy())
        #print(loss["loss"])
        return loss["loss"]
    
    def training_step(self, batch, batch_idx, debug=False):
        if debug and batch_idx==0:
            x = batch
            loss, xloss, candidates_p4, jet_choice = self(x)
            print("training step x",x[0])
            print("training step candidates_p4",candidates_p4[0])
            print("training step jet_choice",jet_choice[0])
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, debug=False):
        if debug and batch_idx==0:
            x = batch
            (c1, c2, c1_out, c2_out, candidates_p4), jet_choice = self(x)
            print("validation step x",x[0])
            print("validation step candidates_p4",candidates_p4[0])
            print("validation step jet_choice",jet_choice[0])
        if batch_idx==0:
            # reset recent performance
            for key in self._recent_perf:
                self._recent_perf[key] = []
        return self.step(batch,batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
        return optimizer

    def learning_rate_scheduler(self):
        # manual learning rate scheduling
        pg = self.trainer.optimizers[0].param_groups[0]
        N = int(0.02 * self.trainer.max_steps) #50e3 # 2% of training steps used for warmup
        # up till N linearly increase (warm-up) the learning rate
        if N!=-1 and self.trainer.global_step < N:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / N)
            pg["lr"] = self.lr * lr_scale # self.lr stays constant
        # beyond exponentially decay
        if N!=-1 and self.trainer.global_step >= N:
            lr_scale = 0.95
            pg["lr"] = self.lr * (lr_scale**(self.trainer.global_step // N))

    def on_train_end(self):
        # merge validation performance for all batches
        perf = {}
        for key in self._recent_perf:
            if self._recent_perf[key][0].shape == (): # if scalar
                perf[key] = np.array(self._recent_perf[key])
            else:
                perf[key] = np.concatenate(self._recent_perf[key], axis=-1)

        # reconstructed mass plot
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()
        _ = ax.hist2d(perf['m0'], perf['m1'], bins=(50,50), range=((0,3000),(0,3000)))
        _ = ax.set_xlabel("mass 0")
        _ = ax.set_ylabel("mass 1")
        self._log_plot(fig, "plot_val_reco_mass")

        # MSE loss plot
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()
        logloss = np.log(perf['loss'])
        _ = ax.hist(logloss, 
                    bins=50, range=(-6,6),
                    histtype='step',
                    density=True,
                    linewidth = 1.5,
                    # linestyle=(linestyle),
                    label=f"dataloader_idx_0")
        _ = ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        _ = ax.set_xlabel("Log(Loss)")
        _ = ax.set_ylabel("Arb. unit")
        self._log_plot(fig, "plot_val_logloss")
        
    
    def loss(self, loss, xloss, candidates_p4):

        # total loss
        l = {}
        l["mse"]         =  torch.mean(loss)
        l["mse_crossed"] =  torch.mean(xloss)

        if self.T ==3:
            isr = candidates_p4[:,2]
            isr_m2 = m2s_from_p4s(isr)
            isr_pt2 = isr[:,1]**2+isr[:,2]**2
            l["ISR_energyT"]  =  torch.mean(torch.sqrt(isr_pt2+isr_m2))*self.loss_config["scale_ISR_loss"]
        
        if 'scale_corr_loss' in self.loss_config:
            m2 = m2s_from_p4s(candidates_p4)
            avgm = (m2[:,0].sqrt() + m2[:,1].sqrt()) / 2.0
            l["corr"] = distance_corr(loss, avgm) * self.loss_config["scale_corr_loss"]

        # get total
        l['loss'] = sum(l.values())
        return l

    def _log_plot(self, fig, name: str):
        tb = self.logger.experiment # type: ignore
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image(name, im)
