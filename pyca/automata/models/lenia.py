import torch,torch.nn,torch.nn.functional as F
import numpy as np
from ..utils.noise_gen import perlin,perlin_fractal
from ..utils import LeniaParams
from showtens import show_image
from ..automaton import Automaton
import pygame,os, random

class MultiLenia(Automaton):
    """
        Multi-channel lenia automaton. A multi-colored GoL-inspired continuous automaton. Introduced by Bert Chan.
    """
    def __init__(self, size, dt, num_channels=3, params: LeniaParams | dict=None, param_path=None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (H,W) of ints, size of the automaton and number of batches
                dt : time-step used when computing the evolution of the automaton
                num_channels : int, number of channels (C) in the automaton
                params : LeniaParams class, or dict of parameters containing the following
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (1,C,C) tensor, mean of growth functions
                    'sigma' : (1,C,C) tensor, standard deviation of the growth functions
                    'beta' :  (1,C,C, # of rings) float, max of the kernel rings 
                    'mu_k' : (1,C,C, # of rings) [0,1.], location of the kernel rings
                    'sigma_k' : (1,C,C, # of rings) float, standard deviation of the kernel rings
                    'weights' : (1,C,C) float, weights for the growth weighted sum
                param_path : path to folder containing saved parameters
                device : str, device 
        """
        super().__init__(size=size)

        self.batch= 1
        self.C = num_channels
        self.device=device

        if(params is None):
            self.params = LeniaParams(batch_size=self.batch, k_size=25,channels=self.C, device=device)
        elif(isinstance(params,dict)):
            self.params = LeniaParams(param_dict=params, device=device)
        else:
            self.params = params.to(device)
        
        self.k_size = self.params['k_size'] # kernel sizes
        self.state = torch.rand(self.batch,self.C,self.h,self.w,device=device)

        self.set_init_fractal() # Fractal perlin init

        self.dt = dt

        self.kernel = torch.zeros((self.k_size,self.k_size),device=device)
        self.fft_kernel = torch.zeros((self.batch,self.C,self.C,self.h,self.w),device=device)
        self.normal_weights = torch.zeros((self.batch,self.C,self.C),device=device)
        self.update_params(self.params)


        self.saved_param_path = param_path
        if(self.saved_param_path is not None):
            self.param_files = [file for file in os.listdir(self.saved_param_path) if file.endswith('.pt')]
            self.num_par = len(self.param_files)
            if(self.num_par>0):
                self.cur_par = random.randint(0,self.num_par-1)
            else:
                self.cur_par = None

        self.to_save_param_path = 'SavedParameters/Lenia'

    def to(self,device):
        self.params.to(device)
        self.kernel = self.kernel.to(device)
        self.normal_weights = self.normal_weights.to(device)
    
    def update_params(self, params, k_size_override = None):
        """
            Updates some or all parameters of the automaton. 
            Changes batch size to match the one of provided params (take mu as reference)

            Args:
                params : LeniaParams or dict, prefer the former
        """
        if(isinstance(params,LeniaParams)):
            params = params.param_dict

        new_dict ={}
        for key in self.params.param_dict.keys():
            new_dict[key] = params.get(key,self.params[key])

        if(k_size_override is not None):
            self.k_size = k_size_override
            if(self.k_size%2==0):
                self.k_size += 1
                print(f'Increased even kernel size to {self.k_size} to be odd')
            new_dict['k_size'] = self.k_size
        self.params = LeniaParams(param_dict=new_dict, device=self.device)

        self.to(self.device)
        self.norm_weights()

        self.batch = self.params.batch_size # update batch size
        self.kernel = self.compute_kernel() # (B,C,C,k_size,k_size)
        self.fft_kernel = self.kernel_to_fft(self.kernel) # (B,C,C,h,w)

    
    def norm_weights(self):
        """
            Normalizes the relative weight sum of the growth functions
            (A_j(t+dt) = A_j(t) + dt G_{ij}w_ij), here we enforce sum_i w_ij = 1
        """
        # Normalizing the weights
        N = self.params.weights.sum(dim=1, keepdim = True) # (B,1,C)
        self.normal_weights = torch.where(N > 1.e-6, self.params.weights/N, 0)

    def get_params(self) -> LeniaParams:
        """
            Get the LeniaParams which defines the automaton
        """
        return self.params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using fractal perlin noise.
            Max wavelength is k_size*1.5, chosen a bit randomly
        """
        self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=self.C,persistence=0.4) 
    
    def set_init_perlin(self,wavelength=None):
        """
            Sets initial state using one-wavelength perlin noise.
            Default wavelength is 2*K_size
        """
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=self.C,black_prop=0.25)
    
    def set_init_circle(self,fractal=False, radius=None):
        if(radius is None):
            radius = self.k_size*3
        if(fractal):
            self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=self.C,persistence=0.4)
        else:
            self.state = perlin((self.batch,self.h,self.w),[self.k_size]*2,
                            device=self.device,num_channels=self.C,black_prop=0.25)
        X,Y = torch.meshgrid(torch.linspace(-self.h//2,self.h//2,self.h,device=self.device),torch.linspace(-self.w//2,self.w//2,self.w,device=self.device))
        R = torch.sqrt(X**2+Y**2)
        self.state = torch.where(R<radius,self.state,torch.zeros_like(self.state,device=self.device))

    def kernel_slice(self, r):
        """
            Given a distance matrix r, computes the kernel of the automaton.
            In other words, compute the kernel 'cross-section' since we always assume
            rotationally symmetric kernel

            Args :
            r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,self.C,self.C,self.params.mu_k.shape[3],-1,-1) #(B,C,C,#of rings,k_size,k_size)

        mu_k = self.params.mu_k[..., None, None] # (B,C,C,#of rings,1,1)
        sigma_k = self.params.sigma_k[..., None, None]# (B,C,C,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,C,C,#of rings,k_size,k_size)
        #print(K.shape)

        beta = self.params.beta[..., None, None] # (B,C,C,#of rings,1,1)
        K = torch.sum(beta*K, dim = 3) #

        
        return K #(B,C,C,k_size, k_size)


    def compute_kernel(self):
        """
            Computes the kernel given the current parameters.
        """
        xyrange = torch.linspace(-1, 1, self.params.k_size).to(self.device)

        X,Y = torch.meshgrid(xyrange, xyrange,indexing='xy') # (k_size,k_size),  axis directions is x increasing to the right, y increasing to the bottom
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,C,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,C,C,k_size,k_size)
    
    def kernel_to_fft(self, K):
        # Pad kernel to match image size
        # For some reason, pad is left;right, top;bottom, (so W,H)
        K = F.pad(K, [0,(self.w-self.params.k_size)] + [0,(self.h-self.params.k_size)]) # (B,C,C,h,w)

        # Center the kernel on the top left corner for fft
        K = K.roll((-(self.params.k_size//2),-(self.params.k_size//2)),dims=(-1,-2)) # (B,C,C,h,w)

        K = torch.fft.fft2(K) # (B,C,C,h,w)

        return K #(B,C,C,h,w)

    def growth(self, u): # u:(B,C,C,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,C,C,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.params.mu[..., None, None] # (B,C,C,1,1)
        sigma = self.params.sigma[...,None,None] # (B,C,C,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(B,C,C,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        U = self.get_fftconv(self.state) # (B,C,C,H,W)

        assert (self.h,self.w) == (U.shape[-2], U.shape[-1])

        weights = self.normal_weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        # Weight normalized growth :
        dx = (self.growth(U)*weights).sum(dim=1) #(B,C,H,W) # G(U)[:,i,j] is contribution of channel i to channel j

        # Apply growth and clamp
        self.state = torch.clamp(self.state + self.dt*dx, 0, 1) # (B,C,H,W)
    
    def get_fftconv(self, state):
        """
            Compute convolution using fft
        """
        state = torch.fft.fft2(state) # (B,C,H,W) fourier transform
        state = state[:,:,None] # (B,1,C,H,W)
        state = state*self.fft_kernel # (B,C,C,H,W), convoluted
        state = torch.fft.ifft2(state) # (B,C,C,H,W), back to spatial domain

        return torch.real(state)


    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1,-2)) # (B,C) mean mass for each color

    def draw(self):
        """
            Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0]

        if(self.C==1):
            toshow = toshow.expand(3,-1,-1)
        elif(self.C==2):
            toshow = torch.cat([toshow,torch.zeros_like(toshow)],dim=0)
        else :
            toshow = toshow[:3,:,:]
    
        self._worldmap = toshow

    def process_event(self, event, camera=None):
        """
        N       -> pick new random parameters
        V       -> vary current parameters slightly
        DEL     -> reset to perlin noise initial state
        M       -> load next saved parameter set
        S       -> save current parameters
        """
        if event.type == pygame.KEYDOWN:
            if(event.key == pygame.K_n):
                """ New random parameters"""
                self.update_params(LeniaParams(k_size=self.params.k_size, channels=self.C, batch_size=self.params.batch_size, device=self.device))
                self.set_init_perlin()
            if(event.key == pygame.K_v):
                """ Variate around parameters"""
                params = self.params.mutate(magnitude=0.1,rate=0.8)
                self.update_params(params)
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.set_init_perlin()
            if(event.key == pygame.K_m):
                if(self.saved_param_path is not None and self.num_par>0):
                    file = os.path.join(self.saved_param_path,self.param_files[self.cur_par])
                    print('loading ', os.path.join(self.saved_param_path,file))
                    self.cur_par = (self.cur_par+1)%self.num_par
                    new_params = LeniaParams(from_file=file)
                    self.update_params(new_params)
                    self.set_init_perlin()
                else:
                    print('No saved parameters')
            if(event.key == pygame.K_s):
                # Save the current parameters:
                self.params.save(self.to_save_param_path)
                print(f'Saved current parameters as {self.params.name}')