from ..automaton import Automaton
import torch, torch.nn as nn
import torch.nn.functional as F
import pygame
from torchenhanced import DevModule, ConfigModule
import math
from easydict import EasyDict
from pathlib import Path

class NeuralCA(Automaton):
    """
        Neural Cellular Automaton. Can be trained to grow any image
        from a 'seed', using only local interactions, and a neural network.
    """

    def __init__(self, size, models_folder, seed=None, device='cpu'):
        """
            Args:
                size : tuple (H,W), size of the automaton
                model_path : str, path to the model (generate with NCAModule.save_model())
                seed : tensor (n_states,h,w), initial state for the automaton. 
                    Note, (h,w) is the size of the seed, which needs not be the
                    size of the screen. If not provided, uses the default all 1's
                    seed of size (n_states,1,1).
                device : str, device on which to run the module
        """
        super().__init__(size)
        self.device = device

        self.models_paths = list(Path(models_folder).rglob('*.pt')) # List of all models in the folder

        self.cur_model = 0 # Current model index
        self.load_model(self.models_paths[self.cur_model]) # Load the first model

        ## TODO : Add the different possible seeds
        ## You could hard-code them, or load them from a folder
        if(seed is None):
            self.seed = torch.ones((self.model.n_states,1,1),device=device)
        else:
            self.seed = seed
        
        self.seed_size = self.seed.shape[1:]
        self.state = torch.zeros((1,self.model.n_states,size[0],size[1]),device=device)

        ## TODO : Potentially need modification if you modify the method
        self.insert_seed(size[0]//2,size[1]//2)

        self.brush_size = 4
        self.m_pos = EasyDict(x=0,y=0)
        self.hgrid, self.wgrid = torch.meshgrid(torch.arange(size[0],device=device),torch.arange(size[1],device=device))

    def load_model(self, model_path):
        """
            Loads the model from model_path.
        """
        # if('device' in model_data['config'].keys()):
        #     model_data['config']['device'] = device
        model_data = torch.load(model_path,map_location=self.device)

        self.model = NCAModule(**model_data['config'])
        self.model.load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad=False

    @torch.no_grad()
    def step(self):
        """
            Step the automaton one step forward.
        """
        self.state = self.model(self.state)    

    def draw(self):
        """
            Draws the automaton state on the screen.
            Might add Blend background in the future
        """
        pic = torch.clamp(self.model.state_to_argb(self.state).squeeze(0),0,1) # (4,H,W)
        
        # Convert to RGB
        pic = pic[:3]*pic[3:] # Blend assuming black background

        self._worldmap = pic # (3,H,W)
        # Draw brush
        brush_mask = self.get_brush_slice(self.m_pos.x,self.m_pos.y)
        self._worldmap = torch.clamp(self._worldmap+torch.where(brush_mask[None],torch.tensor([.2,.2,0.],device=self.device)[:,None,None],0),min=0,max=1)   

    def process_event(self, event, camera=None):
        """
        DELETE              -> resets the automaton
        R                   -> randomize NCA parameters
        LEFT CLICK + DRAG   -> erase cells
        RIGHT CLICK         -> insert seed at cursor position
        SCROLL WHEEL        -> change brush size
        M                   -> load next trained model
        """
        mouse = self.get_mouse_state(camera)
        #Update mouse position, to have it for drawing
        self.m_pos.x = mouse.x
        self.m_pos.y = mouse.y

        ## TODO : potentially add the seed selction with arrow keys ?
        if(event.type == pygame.KEYDOWN):
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.reset()
            if event.key == pygame.K_r:
                self.randomize()
            if event.key == pygame.K_m:
                self.cur_model = (self.cur_model+1)%len(self.models_paths)
                self.load_model(self.models_paths[self.cur_model])
                print('Selected model : ',self.models_paths[self.cur_model])
        if event.type == pygame.MOUSEBUTTONDOWN :
            if event.button == pygame.BUTTON_RIGHT:
                ## TODO : Add seed according to selected one.
                self.insert_seed(int(self.m_pos.y),int(self.m_pos.x))

        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scroll wheel up
                self.brush_size += 1  # Increase brush size
            elif event.y < 0:  # Scroll wheel down
                self.brush_size -= 1  # Decrease brush size
            self.brush_size = max(1, self.brush_size)

        if event.type == pygame.MOUSEMOTION :
            if(mouse.left):
                brush = self.get_brush_slice(self.m_pos.x,self.m_pos.y)
                self.state=torch.where(brush,0,self.state)

    def get_brush_slice(self, x, y):
        """Gets coordinate slices corresponding to the brush located at x,y"""
        set_mask = (self.hgrid-y)**2 + (self.wgrid-x)**2 < self.brush_size**2
        return set_mask # (H,W)
    
    def insert_seed(self, h, w):
        """
            Inserts twe seed centered at twe position (h,w) in twe state.
        """
        ### TODO : You will need to modify this function, to potentially take an argument
        ##  for the specific seed to insert. It could also be random !
        
        # Only put seed if it fits
        if(h+self.seed_size[0]//2<self.size[0] and w+self.seed_size[1]//2<self.size[1]):
            slice_h, slice_w = self._center_slice(h,self.seed_size[0],self.size[0]), \
                                self._center_slice(w,self.seed_size[1],self.size[1])
            self.state[0,:,slice_h,slice_w] = self.seed
        
    def reset(self):
        """
            Resets the automaton to the seed state.
        """
        self.state = torch.zeros_like(self.state)

    def randomize(self):
        """
            Randomizes the automaton parameters
        """
        self.model.randomize()
        self.reset()

    def _center_slice(self, p, sli,maxi):
        return slice(max(0,p-(sli+1)//2),min(p+sli//2,maxi))


class NCAModule(ConfigModule):
    """
        NCA cellular automaton, implemented as a torch module.
        (use NCA class for interactivity and use as an Automaton class.)
    """
    
    def __init__(self,n_states=12,n_hidden=128, device='cpu'):
        """
            Args:
                n_states : int, number of hidden states
                n_hidden : int, number of hidden units
                kernel_size : int, size of the conv kernel
                                (unused for now)

        """

        super().__init__()


        self.n_states=max(4,n_states) # At least 1 hidden state

        self.sobel = SobelConv(n_states=self.n_states,device=device)

        self.computer = nn.Sequential(
            nn.Linear(3*self.n_states,n_hidden,device=device),
            nn.ReLU(),
            nn.Linear(n_hidden,self.n_states,device=device)) # Fully connected that given the sobel observations, computes the next state
        
        # Initialize to identity
        nn.init.zeros_(self.computer[2].weight)
        nn.init.zeros_(self.computer[2].bias)

        print(f"NCA with {self.paranum} parameters")

        self.to(device)
    
    def randomize(self):
        """
            Randomizes the parameters of the model
        """

        for p in self.parameters():
            0.1*nn.init.normal_(p,0,1)

    def getlivemask(self,state):
        """
            Returns a mask of live cells
            Live cell : any neighbor (or itself) has A>0.1

            state : tensor (B,n_states,H,W), state

            Returns:
                mask : tensor (B,1,H,W), live cells
        """
        return F.max_pool2d(F.pad(state[:,3:4,:,:],(1,1,1,1),mode='circular'),
                            kernel_size=3,stride=1)>0.1 
    
    def forward(self, state, n_steps=1):
        """
            Args:
                state : tensor (B,n_states,H,W), init state
                n_steps : int, number of steps to run

            Returns:
                state : tensor (B,n_states,H,W), new state
        """
        B,C,H,W = state.shape

        for _ in range(n_steps):
            dx = self.sobel(state) #(B,3*n_states,H,W) PERCEPTION STEP
            dx = dx.permute(0,2,3,1) # (B,H,W,3*n_states)
            dx = self.computer(dx).permute(0,3,1,2) # (B,n_states,H,W) COMPUTATION STEP

            live_before = self.getlivemask(state) # (B,1,H,W) MASK OF LIVE CELLS
            rand_update = torch.rand(B,1,H,W,device=self.device)<=0.5 # Update stochasticity

            state += dx*rand_update # Update the state

            life_mask = live_before & self.getlivemask(state) # Maks of cells alive before AND after step

            state = state*life_mask # Reset dead cells to zero
        
        return torch.nan_to_num(state,nan=1.0,posinf=100.0,neginf=-100.0)
        
    
    @staticmethod
    def state_to_argb(state):
        """
            Converts (B,n_states,H,W) state to (B,4,H,W) ARGB float tensor
        """
        return state[:,:4]
    
    @staticmethod
    def loss(state, target):
        """
            Args:
                state : tensor (B,n_states,H,W), state
                target : tensor (B,4,H,W), ARGB target

            Returns:
                loss : tensor (B,n_states,H,W), loss per pixel
        """

        return F.mse_loss(NCAModule.state_to_argb(state),target,reduction='none')

    def save_model(self,path):
        """
            Saves the model to path, along with configuration. 
            Should be loaded with self.load_model.
        """
        model_dict = {'config' : self.config, 'state_dict' : self.state_dict()}
        torch.save(model_dict,path)
    
    def load_model(self,path):
        """
            Loads the model from path.
        """
        model_dict = torch.load(path,map_location=self.device)
        if(self.config!=model_dict['config']):
            print(f"Warning : Model configuration mismatch!\n \
            Current : {self.config}\nLoaded : {model_dict['config']}")
        
        self.load_state_dict(model_dict['state_dict'])
    

class SobelConv(DevModule):
    """
        Compute depthwise convolutions using the Sobel filters.
    """
    def __init__(self,n_states, device='cpu'):
        """
            n_states : int, number of channels in NCA
        """
        super().__init__(device)
        self.n_states =n_states        

        sobelx= torch.tensor([[-1,0,1],
                              [-2,0,2],
                              [-1,0,1]],dtype=torch.float)[None,None]/8. # (1,1,3,3)
        sobely= torch.tensor([[-1,-2,-1],
                              [0,0,0],
                              [1,2,1]],dtype=torch.float)[None,None]/8. # (1,1,3,3)
        
        self.register_buffer('sobelkern',torch.cat([sobelx.expand(n_states,-1,-1,-1),
                                                      sobely.expand(n_states,-1,-1,-1)],dim=0)) # (2*n_states,1,3,3)
        self.to(device)

    def forward(self,x):
        """
            Takes a NCA state (B,n_chans,H,W) and returns the state concatenated with the sobel filters

            Args:
                x : tensor (B,n_states,H,W)
            
            Returns:
                out : tensor (B,3*n_states,H,W)
        """
        padx = F.pad(x,pad=(1,1,1,1),mode='circular')

        out = torch.cat([
            F.conv2d(padx,weight=self.sobelkern,groups=self.n_states),
            x],dim=1) # (B,3*n_states,H,W)
        
        return out
    

class SamplePool:
    """
        Pool of samples for NCA training.
    """

    def __init__(self, seed: torch.Tensor, pool_size: int=1024, return_device='cpu'):
        """
            Args:
                seed : tensor (n_states,H,W), NCA 'seed', or initial condition
                pool_size : int, size of the pool
                loss_f : function, should return loss given a batch of states (B,n_states,H,W)
                return_device : str, device on which to return samples
        """

        assert len(seed.shape)==3, "Seed should be (n_states,H,W)"
        self.seed = seed # (n_states,H,W)

        self.pool = seed[None].repeat(pool_size,1,1,1) # Initial pool, full of seeds
        self.p_size = pool_size
        self.device = return_device

        self.h,self.w = seed.shape[1:]
        self.nstates = seed.shape[0]
        self.rmax = min(self.h//2,self.w//2)# For corrupting purposes

        self.ygrid, self.xgrid=torch.meshgrid(torch.arange(0,self.h),torch.arange(0,self.w))# For corrupting purposes
        self.ygrid = self.ygrid[None,None] # (1,1,H,W)
        self.xgrid = self.xgrid[None,None] # (1,1,H,W)

        self.seed_mask = self.seed > 0.01 # (n_states,H,W), location where seed is not zero, avoid corruption
    
    @torch.no_grad()
    def sample(self,num_samples:int,replace_num=1,corrupt=False,num_cor=None):
        """
            Sample num_samples elements of the pool.

            args :
            num_samples : size of batch
            replace_num : number of samples to be replaced with seed
            corrupt : if True, will corrupt half the elements of the batch with erasure.
            num_cor : the number of elements to corrupt, if None, will corrupt a third of the batch

            returns : tuple
            batch of elements (n_samples,C,H,W), indices of the elements in the pool
        """
        assert num_samples<self.p_size, f"Pool size too small for {num_samples} batch!"

        randindices = torch.randint(0,self.p_size,(num_samples,)) # Select random indices for the batch

        batch_init = self.pool[randindices] # Index the selected initial states (num_samples,n_states,H,W)
        
        if(corrupt):
            # Corrupt the batch if corrupt is True
            if(num_cor is None):
                num_cor = num_samples//3 # A third of corruptions

            # Random corruption circle radius : 
            r_size = torch.rand((num_cor,))*self.rmax//3+self.rmax//6 # Random size of corruption circle
            r_size = r_size[:,None,None,None] # (num_cor,1,1,1)

            # Random corruption circle center location : 
            r_loc = self.rmax*torch.rand((num_cor,)) # Random 
            theta_loc = torch.rand((num_cor,))*2*torch.pi

            # Compute the center of the corruption circle, given the sampled r_loc and theta_loc:
            x_loc = (r_loc*torch.cos(theta_loc))[:,None,None,None] + self.w//2 # (num_cor,1,1,1)
            y_loc = (r_loc*torch.sin(theta_loc))[:,None,None,None] + self.h//2
            
            corrupt_mask = (self.xgrid-x_loc)**2+(self.ygrid-y_loc)**2<r_size**2 # (num_cor,1,H,W), mask of where to corrupt
            is_seed = torch.isclose(batch_init[:num_cor],self.seed.expand(num_cor,-1,-1,-1)).\
                        reshape(num_cor,-1).all(dim=1)# (num_cor,), True if the element is seed
            
            batch_init[:num_cor]=torch.where(corrupt_mask,0,batch_init[:num_cor]) # Corrupt the batch
            # To prevent corrupting the seed (which makes it impossible to recover), we re-generate the seed if it was corrupted
            batch_init[:num_cor][is_seed]=self.batch_seed(batch_init[:num_cor][is_seed].shape[0]) 

        if(replace_num>0):
            # Randomly reseed some of the elements
            reseed_ind = torch.randperm(num_samples,device=self.device)[:replace_num] # Select some of the indices to reseed 
            batch_init[reseed_ind] = self.batch_seed(replace_num) # Replace some of the initial states with the seed

        return batch_init.to(self.device), randindices.to(self.device) # (num_samples,n_states,H,W), (num_samples,)
    
    def batch_seed(self,num_samples):
        """
            Returns : batch of seeds (n_samples,C,H,W)
        """

        return self.seed.repeat(num_samples,1,1,1)
    
    @torch.no_grad()
    def update(self,indices, batch, batchloss=None, kill=0.1,cutoff=0.06):
        """
            Update the pool at 'indices' with the provided batch.
            If batchloss is also provided, will update only the strongest
            elements, and replace the 'kill' fraction of the weakest with a seed.

            params:
            indices : (num_samples,) indices referring to the pool location of the given batch
            batch : (num_samples,n_states,H,W) of states to be replaced in pool
            batchloss : (num_samples,) of loss of the batch
            kill : bottom kill fraction in terms of loss are reset to seed
            cutoff : loss cutoff, above which the samples are not updated
        """
        cutoff_mask = batchloss>cutoff
        num_bad = cutoff_mask.sum().item() # Number of samples above cutoff

        # Replace by seed if above cutoff
        batch[cutoff_mask] = self.batch_seed(num_bad).to(batch.device)

        if(batchloss is not None): # If we can cull by loss
            _,sortind = torch.sort(batchloss) # Indices of sorted loss, ascending
            num_to_kill = math.ceil(kill*len(sortind)) # Number of samples to kill
            killind = sortind[-num_to_kill:] # Indices of the samples to kill
            keepind = sortind[:-num_to_kill] # Indices of the samples to keep

            self.pool[indices[keepind].to('cpu')] = batch[keepind].to('cpu') # Update the pool with the survivors
            self.pool[indices[killind].to('cpu')] = self.batch_seed(len(killind)).to('cpu')

        else :
            self.pool[indices.to('cpu')] = batch.to('cpu')


