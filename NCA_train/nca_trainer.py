"""
    Helper class to train NCA model
"""


import sys
sys.path.append('..')

from Automata.models.NCA import NCAModule, SamplePool
import torch, os
from tqdm import tqdm
from torchenhanced import Trainer
from torchenhanced.util import gridify,showTens
from torchvision.transforms import transforms
from PIL import Image

import wandb

class NCA_Trainer(Trainer):
    def __init__(self, model: NCAModule, tar_image:torch.Tensor, frame_run:int = 64, frame_delta:int=32, run_name:str=None, optim = None, scheduler = None, *, save_loc=None, device: str = 'cpu'):
        """
            NCA trainer class. Uses pixel of ones seed by default.
            Parameters:
                model : NCAModule to train
                tar_image : (4,h,w) Target image to train the model on
                frame_run : number of frames to evolve the NCA before evaluation
                frame_delta : Random variations max on frame_run. Introducing stochasticity encourage model to stabilize
                run_name : Name of the run, for wandb logging
                optim : Optimizer to use
                scheduler : Learning rate scheduler to use
                save_loc : Directory to save checkpoints
                device : Device to run the model on
        """
        super().__init__(model, optim, scheduler, save_loc=save_loc, device=device, run_name=run_name, project_name='NCANew')
        self.model = model.to(self.device) # just for auto-completion
        self.tar_image = tar_image.to(self.device)[None] # (1,4,H,W) Target image padded with 4 pixels

        self.world_size = (tar_image.shape[1],tar_image.shape[2]) # Add padding to the target image
        self.frame_run = frame_run # Number of frames to evolve the NCA before evaluation
        seed = torch.zeros(self.model.n_states,*self.world_size) 
        seed[:,self.world_size[0]//2,self.world_size[1]//2] = 1
        self.pool = SamplePool(seed, return_device = self.device) # Pool of samples for training

    

    def train_steps(self, steps, batch_size,*, save_every=50, step_log:int=None):
        """ 
            Train the model for a number of steps

            Parameters:
                steps : int
                    Number of steps to train
                batch_size : int
                    Batch size
                save_every : int
                    Save a checkpoint every 'save_every' steps
                step_log : int
                    Log the loss every 'step_log' steps
        """

        self._init_logger()
        self.logger.define_metric("*", step_metric='steps')

        self.model.train()

        self.step_log=step_log
        self.step_loss=[]

        steps_completed=False
        
        numsteps=0

        load_bar = tqdm(total=steps, desc='Training', unit='step')
        while not steps_completed:
            self.do_step_log = numsteps % self.step_log == 0 if self.step_log is not None else False
            batch, indices = self.pool.sample(num_samples=batch_size, replace_num=batch_size//3, corrupt=True)
            rand_evo = torch.randint(self.frame_run,self.frame_run+32,(1,)).item()

            state = torch.clone(batch)
            for _ in range(rand_evo):
                state = self.model(state)
            
            loss = self.model.loss(state, self.tar_image.expand(batch_size,-1,-1,-1)).mean(dim=(1,2,3)) # (B,) loss per sample
            loss.mean().backward()

            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optim.step()
            self.optim.zero_grad()


            with torch.no_grad():
                self.pool.update(indices, state, batchloss=loss.detach())
                
                self.step_loss.append(loss.mean().item())

                if(self.do_step_log):
                    self.logger.log({'loss/train_step':sum(self.step_loss)/len(self.step_loss)},commit=False)
                    before_after = torch.cat((batch[0:4],state[0:4]),dim=0) # (8,C,H,W)
                    before_after = gridify(self.to_rgb(before_after), out_size=400, columns=4) # (3,H',W') grid of images
                    # showTens(before_after.cpu())
                    before_after = wandb.Image(before_after.permute(1,2,0).cpu().numpy(), caption=f'Up : Before, Down : After')
                    
                    self.logger.log({'evolved' : before_after},commit=False)

                    self._update_x_axis()
                    self.step_loss=[]
            
            self.steps_done+=1
            self.batches+=1
            self.epochs+=1/self.pool.p_size
            load_bar.update(1)

            numsteps+=1
            self._save_and_backup(numsteps, save_every=save_every, backup_every=float('inf'))

            if(numsteps>steps):
                steps_completed=True

    def to_rgb(self, state:torch.Tensor, bg_color:int=0):
        """
            Convert a state tensor to an RGB tensor
        """
        argb = torch.clamp(self.model.state_to_argb(state),min=0,max=1) # (B,4,H,W) ARGB tensor
        return argb[:,:3]*argb[:,3:]+torch.full_like(argb[:,:3],fill_value=bg_color)*(1-argb[:,3:]) # (B,3,H,W) RGB tensor

    def _save_and_backup(self, curstep, save_every, backup_every):
        os.makedirs(self.save_loc,exist_ok=True)
        self.model.save_model(self.save_loc+'/latestNCA.pt')

        return super()._save_and_backup(curstep, save_every, backup_every)


def prepare_img(img_path:str, tarsize:tuple, pad:int=4):
    # Prepare targets and seeds
    target = Image.open(img_path).convert('RGBA') # (C,H,W)

    w,h = target.size

    transfo = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(tarsize),
                                  transforms.Pad(pad,fill=0)])

    return transfo(target)
