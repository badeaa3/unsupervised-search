import torch
import pytorch_lightning as pl
import model_blocks 
from model_blocks import ms_from_p4s, m2s_from_p4s
from math import exp

class StepLightning(pl.LightningModule):

    def __init__(self,
                 encoder_config = {},
                 loss_config = {},
                 lr = 1e-3,
                 update_learning_rate = True,
                 tau_annealing = True,
                 weights = None,
		 L2=0,
                 device="cpu"):
        super().__init__()

        self.Encoder = model_blocks.Encoder(device=device, **encoder_config)
        self.encoder_config = encoder_config
        self.loss_config = loss_config
        self.lr = lr
        self.update_learning_rate = update_learning_rate
        self.tau_annealing = tau_annealing
        self.T = encoder_config["out_dim"]
        if not hasattr(self.T, '__iter__'):
            self.T = [self.T]*encoder_config["attn_blocks_n"]
        else:
            assert len(self.T) == encoder_config["attn_blocks_n"]
        self.maxT = max(self.T)
        self.do_vae = encoder_config.get("do_vae",False)
        self.L2 = L2

        # use the weights hyperparameters
        if weights: 
            ckpt = torch.load(weights,map_location=self.device)
            self.load_state_dict(ckpt["state_dict"])
        
        self.save_hyperparameters()

    def forward(self, x, tau=0.1):

        # create covariant angular piece [eta, cos(phi), sin(phi)]
        w = torch.stack([x[:,:,1], x[:,:,2], x[:,:,3]],-1)

        # create padding mask with True where padded
        mask = (x[:,:,0] == 0).bool()
        loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice, masses = self.Encoder(x, w, mask, tau)
        return loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice, masses
        
    def step(self, batch, batch_idx, version, dataloader_idx=0):
        
        # train update learning rate
        if version == "train" and dataloader_idx==0:
            if self.update_learning_rate:
                self.learning_rate_scheduler()

        tau = self.encoder_config["gumbel_softmax_config"]["tau"]
        if self.tau_annealing:
            tau = tau*(0.1 + 0.9*exp(-self.trainer.global_step/100))
            #self.encoder_config["gumbel_softmax_config"]["tau"] *= 1-1./self.trainer.max_steps #converges to 0.36*original tau

        # forward pass
        x = batch
        loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice, masses = self(x,tau)

        # compute loss
        loss = self.loss(loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice)

        if version == "train" and dataloader_idx==0:
            self.log("tau", tau, prog_bar=True, on_step=True)
            pg = self.trainer.optimizers[0].param_groups[0]
            self.log("lr", pg["lr"], prog_bar=True, on_step=True)
        # log the loss
        if dataloader_idx==0:
            for key, val in loss.items():
                self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=(version=="train"))
        
        #print(loss["loss"])
        return loss["loss"]
    
    def training_step(self, batch, batch_idx, debug=False):
        #debug=True
        if debug and batch_idx==0:
            x = batch
            loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice, (masses, c0_in, c1_in, c0_latent_rep, c1_latent_rep, c0_out, c1_out) = self(x)
            print("training step x",x[0])
            print("training step candidates_p4",candidates_p4[0])
            print("training step jet_choice",jet_choice[0])
            print("training step c0_in",c0_in[0])
            print("training step c0_out",c0_out[0])
            print("training step c1_in",c1_in[0])
            print("training step c1_out",c1_out[0])
            print("training step c0_latent_rep",c0_latent_rep[0])
            print("training step c1_latent_rep",c1_latent_rep[0])
            print("-")
            print("-")
            print("-")
            print("-")
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx, debug=False):
        #debug=True
        x = batch
        loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice, (masses, c0_in, c1_in, c0_latent_rep, c1_latent_rep, c0_out, c1_out) = self(x)
        if debug and batch_idx==0:
            #(c0, c0_latent, c0_out) = candidates_p4
            print("validation step x",x[0])
            print("validation step candidates_p4",candidates_p4[0])
            #print("validation step candidates_p4",c0[0],c0_latent[0],c0_out[0])
            print("validation step jet_choice",jet_choice[0])
            print("validation step c0_in",c0_in[0])
            print("validation step c0_out",c0_out[0])
            print("validation step c1_in",c1_in[0])
            print("validation step c1_out",c1_out[0])
            print("validation step c0_latent_rep",c0_latent_rep[0])
            print("validation step c1_latent_rep",c1_latent_rep[0])
            print("-")
            print("-")
            print("-")
            print("-")
        if dataloader_idx==0:
            return self.step(batch,batch_idx,"val")
        else:
            masses = ms_from_p4s(candidates_p4)
            mavg = (masses[0]+masses[1])/2
            if dataloader_idx==1:
                mystd = torch.sum((mavg - 1100)**2)
            else:
                mystd = torch.sum((mavg - 1500)**2)
            self.log(f"std_{dataloader_idx}", mystd, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
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
    
    def loss(self, loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice):

        # total loss
        l = {}
        if self.do_vae:
            split_size = mu_logvar.shape[-1]//2
            mu = mu_logvar[...,:split_size]
            log_var = mu_logvar[...,split_size:]
            kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = (1,2))
            l["kld"]         =  torch.mean(kld_loss)*self.loss_config["scale_kld_loss"]
        if self.L2:
            l2_crit = torch.nn.MSELoss(size_average=False)
            reg_loss = 0
            for param in self.Encoder.ae_in.parameters():
                reg_loss += l2_crit(param, torch.zeros_like(param))
            for param in self.Encoder.ae_out.parameters():
                reg_loss += l2_crit(param, torch.zeros_like(param))
            l["L2"]= self.L2 * reg_loss

        l["mse"]         =  torch.mean(loss)*self.loss_config.get("scale_reco_loss",1)
        l["mse_crossed"] =  torch.mean(xloss)*self.loss_config["scale_latent_loss"]
        l["mse_rand"]    =  torch.mean(torch.clamp(100+xloss-randloss, min=0))*self.loss_config["scale_random_loss"]
        #l["jet_choice"]  =  -(torch.mean(torch.std(jet_choice, dim=1))+torch.mean(torch.std(jet_choice, dim=2)))*100
        #l["jet_choice"]  =  -(torch.mean(torch.std(jet_choice, dim=1)))*1e0

        if self.maxT ==3:
            isr = candidates_p4[:,2]
            isr_m2 = m2s_from_p4s(isr)
            isr_pt2 = isr[:,1]**2+isr[:,2]**2
            l["ISR_energyT"]  =  torch.mean(torch.sqrt(isr_pt2+isr_m2))*self.loss_config["scale_ISR_loss"]
            #l["ISR_energyT"]  =  torch.mean(isr)*self.loss_config["scale_ISR_loss"]

        # get total
        #print(l)
        l['loss'] = sum(l.values())
        #print({k:v/l['loss'] for k,v in l.items()})
        return l
