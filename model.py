import torch
import pytorch_lightning as pl
import model_blocks 

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
        ae_out, jet_choice, scores, interm_masses = self.Encoder(x, w, mask)
        return ae_out, jet_choice, scores, interm_masses
        
    def step(self, batch, batch_idx, version):
        
        # train update learning rate
        if version == "train" and self.update_learning_rate:
            self.learning_rate_scheduler()
        pg = self.trainer.optimizers[0].param_groups[0]
        self.log("lr", pg["lr"], prog_bar=True, on_step=True)

        if version == "train" and self.tau_annealing:
            self.encoder_config["gumble_softmax_config"]["tau"] *= 1-1./self.trainer.max_steps #converges to 0.36
            self.log("tau", self.encoder_config["gumble_softmax_config"]["tau"], prog_bar=True, on_step=True)

        # forward pass
        x = batch
        (c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4), jet_choice, scores, interm_masses = self(x)
        hard_jet_choice = torch.argmax(jet_choice,dim=-1)
        count_ISR = torch.mean(torch.sum(hard_jet_choice==self.Encoder.T-3, -1).float())
        count_g1  = torch.mean(torch.sum(hard_jet_choice==self.Encoder.T-2, -1).float())
        count_g2  = torch.mean(torch.sum(hard_jet_choice==self.Encoder.T-1, -1).float())
        count_gdiff  = torch.mean(torch.abs(torch.sum(hard_jet_choice==self.Encoder.T-2, -1).float()-torch.sum(hard_jet_choice==self.Encoder.T-1, -1).float()))
        self.log("count_ISR", count_ISR)
        self.log("count_g1", count_g1)
        self.log("count_g2", count_g2)
        self.log("count_gdiff", count_gdiff)
        mass1_in = torch.mean(c1[:,-1])
        mass2_in = torch.mean(c2[:,-1])
        mass1_out = torch.mean(c1_out[:,-1])
        mass2_out = torch.mean(c2_out[:,-1])
        self.log("mass1_in", mass1_in)
        self.log("mass2_in", mass2_in)
        self.log("mass1_out", mass1_out)
        self.log("mass2_out", mass2_out)
        self.log("massdiff", torch.mean((c1[:,-1]-c2[:,-1])**2))
        
        # compute loss
        loss = self.loss(c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4, interm_masses)

        # log the loss
        for key, val in loss.items():
            self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=True)
        
        #print(loss["loss"])
        return loss["loss"]
    
    def training_step(self, batch, batch_idx):
        if batch_idx==0:
            x = batch
            (c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4), jet_choice, scores, interm_masses = self(x)
            print("training step c1",c1[0])
            print("training step c2",c2[0])
            print("training step c1_out",c1_out[0])
            print("training step c2_out",c2_out[0])
            print("training step c1random",c1random[0])
            print("training step c2random",c2random[0])
            print("training step c1random_out",c1random_out[0])
            print("training step c2random_out",c2random_out[0])
            print("training step cp4",cp4[0])
            print("training step jet_choice",jet_choice[0])
            print("training step scores",scores[0])
            print("training step interm_masses",interm_masses[0],interm_masses.shape)
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        if batch_idx==0:
            x = batch
            (c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4), jet_choice, scores, interm_masses = self(x)
            print("validation step x",x[0])
            print("validation step c1",c1[0])
            print("validation step c2",c2[0])
            print("validation step c1_out",c1_out[0])
            print("validation step c2_out",c2_out[0])
            print("validation step c1random",c1random[0])
            print("validation step c2random",c2random[0])
            print("validation step c1random_out",c1random_out[0])
            print("validation step c2random_out",c2random_out[0])
            print("validation step jet_choice",jet_choice[0])
            print("validation step scores",scores[0])
            print("validation step interm_masses",interm_masses[0],interm_masses.shape)
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
    
    def loss(self, c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4, interm_masses):

        ''' 
        cout/cin = [B, E]
        '''

        # total loss
        l = {}
        l["mse"]         =  torch.mean((c1_out-c1)**2 + (c2_out-c2)**2)
        l["mse_crossed"] =  torch.mean((c1_out-c2)**2 + (c2_out-c1)**2)
        l["ISR_energy"]  =  torch.mean(cp4[:,0,0])/10
        #l["gluino_pt2"]  =  -torch.mean(torch.sqrt(cp4[:,-1,1]**2+cp4[:,-1,2]**2)+torch.sqrt(cp4[:,-2,1]**2+cp4[:,-2,2]**2))
        #l["massdiff"]  =  torch.mean((c1[:,-1]-c2[:,-1])**2)/10
        #l["interm_massdiff"]  =  torch.mean((interm_masses[:,-1]-interm_masses[:,-2])**2)
        #l["mse_random"]  =  torch.mean((c1random_out-c1random)**2 + (c2random_out-c2random)**2)
        #l["mse_negative"]  = -torch.mean((c1random_out-c1)**2 + (c2random_out-c2)**2 + (c1random_out-c2)**2 + (c2random_out-c1)**2) #negative, maximize difference to random
        #l["mse_negative"] *= self.loss_config["scale_random_loss"]

        # get total
        l['loss'] = sum(l.values())
        if l['loss'].isnan():
            sys.exit(1)
        return l
