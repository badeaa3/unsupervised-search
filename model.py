import torch
import pytorch_lightning as pl
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

    def forward(self, x):

        # create covariant angular piece [eta, cos(phi), sin(phi)]
        w = torch.stack([x[:,:,1], x[:,:,2], x[:,:,3]],-1)

        # create padding mask with True where padded
        mask = (x[:,:,0] == 0).bool()
        c1, c2, c1_out, c2_out, c1random_out, c2random_out = self.Encoder(x, w, mask)
        return c1, c2, c1_out, c2_out, c1random_out, c2random_out
        
    def step(self, batch, batch_idx, version):
        
        # train update learning rate
        if version == "train" and self.update_learning_rate:
            self.learning_rate_scheduler()
        pg = self.trainer.optimizers[0].param_groups[0]
        self.log("lr", pg["lr"], prog_bar=True, on_step=True)

        # forward pass
        x = batch
        c1, c2, c1_out, c2_out, c1random_out, c2random_out = self(x)
        
        # compute loss
        loss = self.loss(c1, c2, c1_out, c2_out, c1random_out, c2random_out)

        # log the loss
        for key, val in loss.items():
            self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=True)
        
        print(loss["loss"])
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
    
    def loss(self, c1, c2, c1_out, c2_out, c1random_out, c2random_out):

        ''' 
        cout/cin = [B, E]
        '''

        # total loss
        l = {}
        l["mse"]         =  torch.mean((c1_out-c1)**2 + (c2_out-c2)**2)
        l["mse_crossed"] =  torch.mean((c1_out-c2)**2 + (c2_out-c1)**2)
        l["mse_random"]  = -torch.mean((c1random_out-c1)**2 + (c2random_out-c2)**2 + (c1random_out-c2)**2 + (c2random_out-c1)**2) #negative, maximize difference to random

        # get total
        l['loss'] = sum(l.values())
        return l
