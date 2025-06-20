import torch.optim.lr_scheduler as lrsched, os
from torch.optim import Optimizer, lr_scheduler as lrsched
from torchenhanced import Trainer,ConfigModule
from torch.utils.data import DataLoader
import wandb,json, pathlib
import torch, torch.nn.functional as F
from ...DataGen import FastGPTDataset
from ..flexigpt import FlexiGPT


class FlexiGPT_Trainer(Trainer):
    """
        Trainer class for SimGPT.
    """

    def __init__(self, model: FlexiGPT, dataset: FastGPTDataset, val_epochs:int=0, mse : bool= True,
                 optim: Optimizer = None, scheduler: lrsched._LRScheduler = None, 
                save_loc=None, device='cpu', run_name=None, project_name='FlexiGPT',
                v_path='',run_config={}):
        """
        Args :
            model : DevModule, model to be trained
            dataset : CAVidDataset, (infinite) dataset to be trained on
            val_epoch : Number of batches in the validation epoch. If 0, skips validation.
            pred_distance : frame distance at which prediction starts. Eg,
            if it is 2, then given the frames 0...,n will try to predict n+2.
            mse : If true, uses MSE, otherwise, cross entropy. Only works if images have values in 0 and 1.
            noise_prob : flips pixels with noise_prob probability
            blur_sigma : How much to blur the input images. If None, no blur.
            mse : if true, use mse loss, otherwise cross-entropy.
            rest : see torchenhanced.Trainer (hopefully doc transfers)
        """
        super().__init__(model=model, optim=optim, scheduler=scheduler, save_loc=save_loc, 
                         device=device, run_name=run_name, project_name=project_name, run_config=run_config)
        
        self.dataset = dataset # Dataset might add noise/change init condition

        al_x = self.dataset.al_x
        al_y = self.dataset.al_y
        val_x = list(set([i for i in range(512)]).difference(set(al_x.tolist()))) # Disallowed x values
        val_y = list(set([i for i in range(512)]).difference(set(al_y.tolist()))) # Disallowed y values

        if(len(val_x)==0):
            print('All rules used in training, validation mirroring training.')
            val_x = al_x.tolist()
            val_y = al_y.tolist()

        self.validataset = FastGPTDataset(self.dataset.img_size,self.dataset.num_frames,batch_size=self.dataset.batch_size,gen_device=self.dataset.gen_device,
                                                allowed_x=val_x,allowed_y=val_y,epoch_length=val_epochs,frame_pred=self.dataset.fp,pred_distance=self.dataset.fd,
                                                noise_prob=self.dataset.noise_prob,blur=self.dataset.blur,skip_frames=self.dataset.skip_frames,
                                                wait_time=self.dataset.wait_time,backwards=self.dataset.backwards, vary_init_size=self.dataset.vary_init)
        
        self.mse = mse # True if using mse

        # For advanced logging purposes
        self.lastframeloss = []
        self.lastframeBCEloss = []

        self.val_rules = {}

        if os.path.exists(v_path):
            print('YO v_path : ', v_path, ' exists, loading validation rules')
            for file in os.listdir(v_path):
                with open(os.path.join(v_path,file), 'r') as f :
                    self.val_rules[file[:-5]] = json.load(f) # Should be a list of tuples of int
            self.full_valid=True
            print('Found val rules :', self.val_rules.keys())
        else :
            print('YO v_path : ', v_path, ' does not exist, no validation rules loaded')
            self.full_valid=False 
    
    def train_init(self,freeze=False, enc_dec_path=None):
        """
            Initialize training, and freeze encoder-decoder if needed.

            freeze : if True, freeze encoder-decoder
            load_from : if not None, load encoder-decoder from this location
        """

        if(enc_dec_path is not None):
            self.model.load_enc_dec(enc_dec_path)
            print('LOADED ENCODER-DECODER !')
            
        if(freeze):
            self.model.freeze_enc_dec()
            print('FROZEN ENCODER-DECODER')

    def save_state(self, epoch: int = None, suffix=''):
        super().save_state(suffix=suffix, epoch=epoch)
        self.model.save_enc_dec(os.path.join(self.save_loc,f'{self.run_name}_enc_dec.state'))
    
    def get_loaders(self, batch_size,num_workers=0):
        self.dataset.batch_size = batch_size
        self.validataset.batch_size = batch_size

        print('LEN DATASET : ', len(self.dataset))
        # None batch_size, because dataset returns batches of data already
        dataloader = DataLoader(self.dataset,batch_size=None,num_workers=0)
        print('LEN DATALOADER : ', len(dataloader))
        if(self.validataset.epoch_length==0):
            validloader=None
        else :
            validloader = DataLoader(self.validataset,batch_size=None,num_workers=0)
            print('LEN VALIDLOADER : ', len(validloader))


        return dataloader, validloader
    

    def process_batch(self, batch_data, **kwargs):
        vid_input, vid_answer= batch_data # FastGPTDataset directly outputs the input and the ground-truth, correctly processed.
        vid_input= vid_input.to(self.device)
        vid_answer= vid_answer.to(self.device)

        pred = self.model(vid_input) # (B,T,frame_pred,C,H,W), predict on preprocessed data

        if(self.mse):
            loss = F.mse_loss(pred,vid_answer)
        else :
            loss = F.binary_cross_entropy_with_logits(pred,vid_answer)

        with torch.no_grad():
            if(self.do_step_log):
                fp = pred.shape[2]
                # Aggregate only the last frame prediction (worst case scenario). I also include the first few frames which are not last, oh well.
                true = torch.cat([vid_answer[:4,0,:fp],vid_answer[:4,1:,fp-1]],dim=1) #(4,T + pred_frame-1,1,H,W)
                false = torch.cat([pred[:4,0,:fp],pred[:4,1:,fp-1]],dim=1) # (4,T + pred_frame-1,1,H,W)

                if(not self.mse): # Need to adjust logits
                    false = F.sigmoid(false)

                true = true.detach().cpu().expand(-1,-1,3,-1,-1) # (4,T+pred_frame-1,3,H,W)
                false = false.detach().cpu().repeat(1,1,3,1,1) # (4,T+pred_frame-1,3,H,W)
                false[:,:,0]=0.6*false[:,:,0]
                combined = torch.stack([true,false],dim=1)
                combined = torch.clamp(torch.flatten(combined,0,1),min=0,max=1) # (8,T,3,H,W), true and false alternating

                wandb.log({'Vids/ComparePrediction' : wandb.Video((combined[:]*255).to(torch.uint8).numpy(), fps=2, format="gif")},commit=False)
                
                plotpast = vid_input[:4].detach().cpu().expand(-1,-1,3,-1,-1)
                wandb.log({'Vids/InputGiven ' : wandb.Video((plotpast*255).to(torch.uint8).numpy(), fps=2,format="gif")},commit=False)
                # log lr
                wandb.log({'lr' : self.scheduler.get_last_lr()[0]},commit=False)

        return loss
    
    def epoch_log(self):
        wandb.log({'lr' : self.scheduler.get_last_lr()[0]},commit=False)

    @torch.no_grad()
    def process_batch_valid(self, batch_data, **kwargs):
        vid, future= batch_data 

        vid= vid.to(self.device)
        future= future.to(self.device)

        pred = self.model(vid) # (B,T,frame_pred,C,H,W), predict on preprocessed data
        

        if(self.mse):
            loss = F.mse_loss(pred,future)
        else :
            loss = F.binary_cross_entropy_with_logits(pred,future)
        
        if(self.mse):
            self.lastframeloss.append(F.mse_loss(pred[:,:,-1],future[:,:,-1]))
        else :
            self.lastframeloss.append(F.binary_cross_entropy_with_logits(pred[:,:,-1],future[:,:,-1]))
    
        
        return loss
    
    @torch.no_grad()
    def valid_log(self):
        wandb.log({'valid_datasets/lastframeloss' : sum(self.lastframeloss)/len(self.lastframeloss)},commit= False)
        self.lastframeloss=[]
        print('FULL VALID : ',self.full_valid)
        if(self.full_valid):
            val_losses = self.score_valid_data()
            print('MANAMHJEEEEEF : ', val_losses)

            for val_name,val_loss in val_losses.items():
                wandb.log({f'valid_datasets/{val_name}' :val_loss}, commit=False)
        



    @torch.no_grad()
    def score_valid_data(self, max_rules=10000):
        """
            Return average loss on a max_rules chunk of the saved validation datasets.
        """

        losses = {}
        for rule_name,rule_list in self.val_rules.items():
            ind_choice = torch.randperm(len(rule_list)) # Indices of choosen periodic rules
            
            rule_batch_size = self.dataset.batch_size
            
            rules = torch.tensor(rule_list,device=self.device)[ind_choice[:max_rules]] # chosen 500 rules to test on
            rule_ex=1 # number of examples per rule

            error = []
            for i in range(max(1,len(rules)//rule_batch_size)):
                x = rules[i*rule_batch_size:min((i+1)*rule_batch_size,len(rules)),0] # (B,) x tensor
                y = rules[i*rule_batch_size:min((i+1)*rule_batch_size,len(rules)),1] # (B,) y tensor

                n_rules = x.shape[0]

                vid, fut = self.dataset.get_batch_rules(n_examples = rule_ex, rules_x=x, rules_y = y) # (n_rules,n_ex,T,C,H,W), (n_rules,n_ex,T,fp,C,H,W)

                _,_,T,C,H,W = vid.shape
                _,_,Tf,fp,_,_,_ = fut.shape

                vid = vid.reshape(n_rules*rule_ex,T,C,H,W)
                fut = fut.reshape(n_rules*rule_ex,Tf,fp,C,H,W)

                pred = self.model(vid) # (n_rules*n_ex,T,frame_pred,C,H,W)

                if(self.mse):
                    error.append(F.mse_loss(pred,fut,reduction='mean'))
                else :
                    error.append(F.binary_cross_entropy_with_logits(pred,fut,reduction='mean')) 
            if(len(error)>0):
                losses[rule_name] = sum(error)/len(error)
            else :
                raise ValueError("Problem in validation, loss array empty")

        return losses # Dict with avg loss for each class
