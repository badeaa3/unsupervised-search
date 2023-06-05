import torch
import torch.nn as nn
import pytorch_lightning as pl
import itertools
import model_blocks 

class StepLightning(pl.LightningModule):

    def __init__(self,
                 encoder_config = {},
                 loss_config = {},
                 lr = 1e-3,
                 update_learning_rate = True,
                 weights = None):
        super().__init__()

        self.Encoder = model_blocks.Encoder(**encoder_config)
        self.loss_config = loss_config
        self.lr = lr
        self.update_learning_rate = update_learning_rate

        # use the weights hyperparameters
        if weights: 
            ckpt = torch.load(weights,map_location=self.device)
            self.load_state_dict(ckpt["state_dict"])
        
        self.save_hyperparameters()

    def forward(self, x, y=None):

        # create covariant angular piece [eta, cos(phi), sin(phi)]
        w = torch.stack([x[:,:,1], x[:,:,2], x[:,:,3]],-1)

        # create padding mask with True where padded
        mask = (x[:,:,0] == 0).bool()
        x, w = self.Encoder(x, w, mask, y, self.mass_loss)
        
        return x, mask
        
    def step(self, batch, batch_idx, version):
        
        # train update learning rate
        if version == "train" and self.update_learning_rate:
            self.learning_rate_scheduler()
        pg = self.trainer.optimizers[0].param_groups[0]
        self.log("lr", pg["lr"], prog_bar=True, on_step=True)

        # forward pass
        x, y  = batch
        y_hat, mask = self(x,y)
        
        # compute loss
        loss = self.loss(y_hat, y, mask)

        # log the loss
        for key, val in loss.items():
            self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=True)
        
        # compute and log accuracy
        acc = self.accuracy(y_hat, y)
        self.log(f"{version}_acc", acc, prog_bar=False, on_step=True)

        return loss["loss"]
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
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
    
    def accuracy(self, y_hat, y):
        
        # target labels
        y_targ = y[:,:-2].long()
        # weights second to last
        weights = y[:,-2]
        # class label always last
        y_class = y[:,-1].bool()
        # flip glu 1 and glu 2
        temp = torch.clone(y_targ)
        temp[y_targ==0] = 1
        temp[y_targ==1] = 0

        # max of prediction
        y_hat_idx = y_hat.argmax(2)
        # compute accuracy
        acc = (((y_hat_idx == y_targ).sum(1) == y_hat_idx.shape[1]) + ((y_hat_idx == temp).sum(1) == y_hat_idx.shape[1])).sum() / y_hat_idx.shape[0]

        return acc

    def mass_loss(self, y_hat, y, mask, lambda_s=1):
        
        # target labels
        y_targ = y[:,:-2].long()
        # weights second to last
        weights = y[:,-2]
        # class label always last
        y_class = y[:,-1].bool()
        
        # mask the target which has -1 for pad
        y_targ[mask] = 0

        # symmetric loss
        l_mass = torch.nn.functional.cross_entropy(y_hat, y_targ, reduction="none")
        l_mass = (~mask).int() * l_mass # apply mask per jet (remember mask=True for ISR)
        l_mass = l_mass.mean(1) # average over event
        # flip glu1 and glu2
        temp = torch.clone(y_targ)
        temp[y_targ==0] = 1
        temp[y_targ==1] = 0
        l_mass_1 = torch.nn.functional.cross_entropy(y_hat, temp, reduction="none")
        l_mass_1 = (~mask).int() * l_mass_1 # apply mask per jet (remember mask=True for ISR)
        l_mass_1 = l_mass_1.mean(1) # average over event
        l_mass = torch.minimum(l_mass, l_mass_1)
        
        # sum loss
        l = {"mass_sig" : float(0), "mass_bkg" : float(0)}
        if y_class.sum() > 0:
            if weights is not None:
                l["mass_sig"] = lambda_s * torch.mean(weights[y_class] * l_mass[y_class])
            else:
                l["mass_sig"] = lambda_s * torch.mean(l_mass[y_class])
        if (~y_class).sum() > 0:
            if weights is not None:
                l["mass_bkg"] = torch.mean(weights[~y_class] * l_mass[~y_class])
            else:
                l["mass_bkg"] = torch.mean(l_mass[~y_class])

        return l

    def loss(self, y_hat, y, mask):
        ''' 
        y = format (B, NJets+1+1) with [label, event weight, signal/background from dsid]
        yhat = [B, NJets, NParent] 
        mask = [B, NJets]
        '''

        # total loss
        l = {}

        # get the mass loss
        l_mass = self.mass_loss(y_hat.transpose(2,1), y, mask)

        # total loss
        l = {
            'aux_mass_loss' : self.loss_config["scale_auxiliary_loss"]*self.Encoder.block_mass_loss,
            'mass_sig' : l_mass['mass_sig'],
            'mass_bkg' : l_mass['mass_bkg'],
        }
        l['loss'] = sum(l.values())

        return l
