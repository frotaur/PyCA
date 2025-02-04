from ..Automaton import Automaton
import torch, torch.nn as nn
import torch.nn.functional as F
import pygame
from torchenhanced import DevModule, ConfigModule
import math


class NCA(Automaton):
    """
        Neural Cellular Automaton. Wrapper for NCAModule, adds visualization
        and interaction.
    """

    def __init__(self, size, model_path, seed=None, device='cpu'):
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

        model_data = torch.load(model_path,map_location=device)
        self.model = NCAModule(**model_data['config'],device=device)
        self.model.load_model(model_path)
        self.model.eval()
        
        for p in self.model.parameters():
            p.requires_grad=False

        if(seed is None):
            self.seed = torch.ones((self.model.n_states,1,1),device=device)
        else:
            self.seed = seed
        
        self.seed_size = self.seed.shape[1:]
        self.state = torch.zeros((1,self.model.n_states,size[0],size[1]),device=device)
        self.insert_seed(size[0]//2,size[1]//2)
        self.device = device


        # Interaction
        self.left_dragging = False
        self.right_dragging = False 

        self.hgrid, self.wgrid = torch.meshgrid(torch.arange(size[0],device=device),torch.arange(size[1],device=device))

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
    
    def process_event(self, event, camera=None):
        """
        DELETE              -> resets the automaton
        F                   -> randomize parameters
        LEFT CLICK + DRAG   -> erase cells
        RIGHT CLICK         -> insert seed at cursor position
        """
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                self.reset()
            if event.key == pygame.K_f:
                self.randomize()
                print('Randomized')
        if event.type == pygame.MOUSEBUTTONDOWN :
            if event.button == pygame.BUTTON_LEFT:  # If left mouse button pressed
                self.left_dragging = True
            if event.button == pygame.BUTTON_RIGHT:
                w,h = camera.convert_mouse_pos(event.pos)
                self.insert_seed(int(h),int(w))

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == pygame.BUTTON_LEFT:  # If left mouse button released
                self.left_dragging = False
        elif event.type == pygame.MOUSEMOTION :
            (w,h) = camera.convert_mouse_pos(event.pos)

            if(self.left_dragging):
                self.state=torch.where((self.hgrid-h)**2+(self.wgrid-w)**2<16,0,self.state)

    def insert_seed(self, h, w):
        """
            Inserts twe seed centered at twe position (h,w) in twe state.
        """
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
        (use NCA for interactivity and use as an Automaton class.)
    """

    def __init__(self,kernel_size=3,n_states=12,n_hidden=128, device='cpu'):
        """
            Args:
                n_states : int, number of hidden states
                n_hidden : int, number of hidden units
                kernel_size : int, size of the conv kernel
                                (unused for now)

        """
        config=dict(kernel_size=kernel_size,n_states=n_states,
                n_hidden=n_hidden)
        super().__init__(config=config,device=device)


        
        self.n_states=max(4,n_states) # At least 1 hidden state

        self.kern = kernel_size

        self.sobel= CABlock(n_states=self.n_states,device=device)

        self.think = nn.Sequential(
            nn.Linear(3*self.n_states,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,self.n_states))
        
        # Initialize to identity
        nn.init.zeros_(self.think[2].weight)
        nn.init.zeros_(self.think[2].bias)

        print(f"NCA with {self.paranum} parameters")

        self.to(device)
    
    def randomize(self):
        """
            Randomizes the parameters of the model
        """

        for p in self.parameters():
            0.1*nn.init.normal_(p,0,1)

            print('bruv')

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
            dx = self.sobel(state) #(B,3*n_states,H,W)
            dx = dx.permute(0,2,3,1) # (B,H,W,3*n_states)
            dx = self.think(dx).permute(0,3,1,2) # (B,n_states,H,W)

            live_before = self.getlivemask(state) # (B,1,H,W)

            rand_update = torch.rand(B,1,H,W,device=self.device)<=0.5 # Update with chance .5

            state += dx*rand_update

            life_mask = live_before & self.getlivemask(state)

            state = state*life_mask # Kill cells that were not alive before and after
        
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
        assert self.config==model_dict['config'], f"Model configuration mismatch!\n \
            Current : {self.config}\nLoaded : {model_dict['config']}"
        
        self.load_state_dict(model_dict['state_dict'])
    

class CABlock(DevModule):
    """
        Compute depthwise convolutions using the sobel filters.
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
        
        # self.register_buffer('sobelx',sobelx.repeat(n_states,1,1,1)) # (n_states,1,3,3)
        # self.register_buffer('sobely',sobely.repeat(n_states,1,1,1)) # (n_states,1,3,3)
        self.register_buffer('sobelkern',torch.cat([sobelx.expand(n_states,-1,-1,-1),
                                                      sobely.expand(n_states,-1,-1,-1)],dim=0)) # (2*n_states,1,3,3)
        self.to(device)

    def forward(self,x):
        """
            Takes state, spits out state :

            x : tensor (B,n_states,H,W)
        """
        padx = F.pad(x,pad=(1,1,1,1),mode='circular')
        # out = torch.cat([
        #     F.conv2d(padx,self.sobelx,groups=self.n_states),
        #     F.conv2d(padx,self.sobely,groups=self.n_states),
        #     x],dim=1) #(B,3*n_states,H,W)

        out = torch.cat([
            F.conv2d(padx,weight=self.sobelkern,groups=self.n_states),
            x],dim=1) # (B,3*n_states,H,W)
        
        return out
    

class SamplePool:
    """
        Pool of samples for training NCA's.
    """

    def __init__(self, seed: torch.Tensor, pool_size: int=1024, loss_f = None, return_device='cpu'):
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
        self.device=return_device

        self.h,self.w = seed.shape[1:]
        self.nstates = seed.shape[0]
        self.rmax = min(self.h//2,self.w//2)# For corrupting purposes

        self.ygrid, self.xgrid=torch.meshgrid(torch.arange(0,self.h),torch.arange(0,self.w))
        self.ygrid = self.ygrid[None,None] # (1,1,H,W)
        self.xgrid = self.xgrid[None,None]

        self.seed_mask = self.seed > 0.01 # (n_states,H,W), location where seed is not zero
        self.loss_f = loss_f
    
    @torch.no_grad()
    def sample(self,num_samples:int,replace_num=1,corrupt=False,num_cor=None):
        """
            Sample num_samples elements of the pool.

            args :
            num_samples : size of batch
            replace_num : number of samples to be replaced with seed
            corrupt : if True, will corrupt half the elements of the batch with erasure.

            returns : tuple
            batch of elements (n_samples,C,H,W), indices of the elements in the pool
        """
        assert num_samples<self.p_size, f"Pool size too small for {num_samples} batch!"

        randindices = torch.randint(0,self.p_size,(num_samples,)) # Select random indices for the batch

        batch_init = self.pool[randindices] # Index the selected initial states (num_samples,n_states,H,W)
        
        if(corrupt):
            if(num_cor is None):
                num_cor = num_samples//3 # A third of corruptions

            # Draw a the corruption circle center by drawing r and theta
            r_size = torch.rand((num_cor,))*self.rmax//3+self.rmax//6 # (num_cor)
            r_loc = (self.rmax-(r_size+.5))*torch.rand((num_cor,))
            theta_loc = torch.rand((num_cor,))*2*torch.pi

            x_loc = (r_loc*torch.cos(theta_loc))[:,None,None,None] + self.w//2# (num_cor,1,1,1)
            y_loc = (r_loc*torch.sin(theta_loc))[:,None,None,None] + self.h//2
            
            radius2 = (self.xgrid-x_loc)**2+(self.ygrid-y_loc)**2 # (num_cor,1,H,W)
            is_seed = torch.isclose(batch_init[:num_cor],self.seed.expand(num_cor,-1,-1,-1)).\
                        reshape(num_cor,-1).all(dim=1)# (num_cor,)
            
            batch_init[:num_cor]=torch.where((radius2<r_size[:,None,None,None]**2),0,batch_init[:num_cor])
            batch_init[:num_cor][is_seed]=self.batch_blank_seed(batch_init[:num_cor][is_seed].shape[0])   # Replace with seed if it was seed, to prevent corruption
    
        if(replace_num>0):
            reseed_ind = torch.randperm(num_samples,device=self.device)[:replace_num] # Select some of the indices to reseed 
            batch_init[reseed_ind] = self.batch_blank_seed(replace_num) # Replace some of the initial states with the seeds, but not changing the species

        return batch_init.to(self.device), randindices.to(self.device) # (num_samples,n_states,H,W),(num_samples),
    
    def batch_blank_seed(self,num_samples):
        """
            Returns : batch of seeds (n_samples,C,H,W)
        """

        return self.seed.repeat(num_samples,1,1,1)
    
    
    def update(self,indices, batch, batchloss=None, kill=0.1,cutoff=0.06):
        """
            Update the pool at 'indices' with the provided batch.
            If batchloss is also provided, will update only the strongest
            elements, and replace the 'kill' fraction of the weakest with a seed.

            params:
            indices : (num_samples,) of ints between 0 and pool_size
            batch : (num_samples,n_states,H,W) of states to be replaced in pool
            batchloss : (num_samples,) of loss of the batch
            kill : bottom kill fraction in terms of loss are reset to seed
            cutoff : loss cutoff, above which the samples are not updated
        """
        cutoff_mask = batchloss>cutoff
        num_bad = cutoff_mask.sum().item()
        # Replace by seed if above cutoff
        batch[cutoff_mask] = self.batch_blank_seed(num_bad).to(batch.device)

        if(batchloss is not None):
            _,sortind = torch.sort(batchloss) # Indices of sorted loss, ascending
            killind = sortind[-math.ceil(kill*len(sortind)):] # Kill the higher loss
            # Very inefficient, replacing twice, TODO : Change this later
            self.pool[indices[sortind].to('cpu')] = batch[sortind].to('cpu')
            self.pool[indices[killind].to('cpu')] = self.batch_blank_seed(len(killind)).to('cpu')

        else :
            self.pool[indices.to('cpu')] = batch.to('cpu')


