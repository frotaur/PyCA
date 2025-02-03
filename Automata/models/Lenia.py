from ..Automaton import Automaton
import torch, math, pygame, os, random
from pyperlin import FractalPerlin2D
import torch.nn.functional as F

class MultiLenia(Automaton):
    """
        Generalized Lenia-like automaton.

        Args :
        size : tuple of ints, size of the automaton
        dt : time-step used when computing the evolution of the automaton
        params : dict of tensors containing the parameters.
        param_path : path to folder containing saved parameters
        state_init : initial state, shape (3,H,W)
        device : str, device 
    """
    
    def __init__(self, size, dt=0.1, params=None, param_path=None, state_init = None,  device='cpu' ):
        super().__init__(size)

        self.dt = dt
        self.device=device   

        if(params is None):
            self.k_size=25
            params = self.gen_params(device)
        
        self.update_params(params)

        self.state = torch.rand((3,self.h,self.w),device=device) # State of the world

        if(state_init is None):
            self.set_init_perlin()
        else:
            self.state = state_init.to(self.device)



        self.param_path = param_path

        if (param_path is not None):
            os.makedirs(param_path,exist_ok=True)
            self.param_files = [file for file in os.listdir(self.param_path) if file.endswith('.pt')]
            self.num_par = len(self.param_files)
            if(self.num_par>0):
                self.cur_par = random.randint(0,self.num_par-1)

    def norm_weights(self):
        # Normalizing the weights
        N = self.weights.sum(dim=0, keepdim = True)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)
    
    def update_params(self, params):
        """
            Updates the parameters of the automaton.
        """
        # Add kernel size
        self.k_size = params['k_size'] # kernel sizes (same for all)
        self.mu=  params['mu'] # mean of the growth functions (3,3)
        self.sigma=  params['sigma'] # standard deviation of the growths functions (3,3)
        self.beta= params['beta'] # max of the kernel rings (3,3,# of rings)
        self.mu_k= params['mu_k']# mean of the kernel gaussians (3,3,# of rings)
        self.sigma_k= params['sigma_k']# standard deviation of the kernel gaussians (3,3,# of rings)
        self.weights= params['weights'] # weigths for the growth weighted sum (3,3)

        self.norm_weights()

        self.kernel = self.gen_kernel() # (3,3,h, w)


    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        # Add output of kernel size
        params = dict(k_size = self.k_size, mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params


    def set_init_perlin(self,wavelength=None):
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((1,self.h,self.w),[wavelength]*2,device=self.device,black_prop=0.25)[0]
    
    def _kernel_slice(self, r): # r : (k_size,k_size)
        """ 
            Helper function.
            Given a distance tensor r, computes the kernel of the automaton.

            Args :
            r : (k_size,k_size), value of the radius for each location around the center of the kernel
        """
        r = r[None, None, None] #(1, 1, 1, k_size, k_size)
        r = r.expand(3,3,self.mu_k[0][0].size()[0],-1,-1) #(3,3,#of rings,k_size,k_size)

        mu_k = self.mu_k[:,:,:, None, None]
        sigma_k = self.sigma_k[:,:,:, None, None]

        # K = torch.exp(-((r-mu_k)/2)**2/sigma_k) #(3,3,#of rings,k_size,k_size)
        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) 
        #print(K.shape)

        beta = self.beta[:,:,:, None, None]

        K = torch.sum(beta*K, dim = 2)

        
        return K #(3,3,k_size, k_size)
    
    def gen_kernel(self):
        """
            Generates the kernel with currents parameters and returns it.

            Returns :
            (3,3,k,k) tensor, the multi channel lenia kernel
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self._kernel_slice(r) #(3,3,k_size,k_size)

        # Normalize the kernel
        summed = torch.sum(K, dim = (2,3), keepdim=True) #(3,3,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(3,3,k,k)
    
    def growth(self, u): 
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (3,3,H,W) tensor of concentrations.

            Returns : 
            (3,3,H,W) tensor of growths
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.mu[:,:, None, None]
        sigma = self.sigma[:,:,None,None]
        mu = mu.expand(-1,-1, self.h, self.w)
        sigma = sigma.expand(-1,-1, self.h, self.w)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(3,3,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """

        # Compute the convolutions.
        # We use torch's Conv2d, we have to make some shenanigans
        # to make the convolutions in all channels at once
        kernel_eff = self.kernel.reshape([9,1,self.k_size,self.k_size])#(9,1,k,k)


        U = F.pad(self.state[None], [(self.k_size-1)//2]*4, mode = 'circular') # (1,3,H+pad,W+pad)
        U = F.conv2d(U, kernel_eff, groups=3).squeeze(0) #(9,H,W)
        U = U.reshape(3,3,self.h,self.w)

        assert (self.h,self.w) == (self.state.shape[1], self.state.shape[2])
 
        weights = self.weights [...,None, None]
        weights = weights.expand(-1, -1, self.h,self.w) #

        # Compute the update to the state
        dx = (self.growth(U)*weights).sum(dim=0) #(3,H,W)
        
        # Update the state
        self.state = torch.clamp( self.state + self.dt*dx, 0, 1) 
    
    def draw(self):
        """
            Draws the worldmap from state.
        """
        
        self._worldmap= self.state # Super simple, just the state, directly   
        
    
    def gen_params(self,device):
        """ Generates parameters which are expected to sometime die, sometime live. Very Heuristic."""
        mu = torch.rand((3,3), device=device)
        sigma = mu/(3*math.sqrt(2*math.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
            

        params = {
            'k_size' : self.k_size, 
            'mu':  mu ,
            'sigma' : sigma,
            'beta' : torch.rand((3,3,3), device=device),
            'mu_k' : torch.rand((3,3,3), device=device),
            'sigma_k' : torch.rand((3,3,3), device=device),
            'weights' : torch.rand((3,3),device = device) # element i, j represents contribution from channel i to channel j
        }

        return params
    
    def process_event(self, event, camera=None):
        """
        CANC    -> resets the automaton
        N       -> pick new random parameters
        V       -> vary current parameters slightly
        Z       -> reset to perlin noise initial state
        M       -> load next saved parameter set
        S       -> save current parameters
        """
        if event.type == pygame.KEYDOWN:
            if(event.key == pygame.K_n):
                """ New random parameters"""
                self.update_params(self.gen_params(self.device))
            if(event.key == pygame.K_v):
                """ Variate around parameters"""
                self.around_params()
            if(event.key == pygame.K_z):
                self.set_init_perlin()
                n_steps=0
            if(event.key == pygame.K_m):
                if(self.param_path is not None):
                    # Load random interesting param
                    if(self.num_par >0):
                        file = os.path.join(self.param_path,self.param_files[self.cur_par])
                        self.cur_par = (self.cur_par+1)%self.num_par
                        self.update_params(torch.load(file, map_location=self.device))
                        print('loaded ', os.path.join(self.param_path,file))
            if(event.key == pygame.K_s):
                # Save the current parameters:
                para = self.get_params()
                name = f'save_mu{para["mu"][0][0][0].item():.2f}_si{para["sigma"][0][0][0].item():.2f}_be{para["beta"][0,0,0,0].item():.2f}'
                name = os.path.join(self.param_path,name+'.pt')
                torch.save(para, name)

    def around_params(self,device):
        """
            Gets parameters which are perturbations around the given set.

            args :
            params : dict of parameters. See LeniaMC for the keys.
        """
        # Make variations proportional to current value
        p = self.get_params()
        p = {
            'k_size' : p['k_size'],
            'mu' : p['mu']*(1 + 0.02*torch.randn((3,3), device=device)),
            'sigma' : torch.clamp(p['sigma']*(1 + 0.02*torch.randn((3,3), device=device)), 0, None),
            'beta' : torch.clamp(p['beta']*(1 + 0.02*torch.randn((3,3,1), device=device)),0,1),
            'mu_k' : p['mu_k']*(1 + 0.02*torch.randn((3,3,1), device=device)),
            'sigma_k' : torch.clamp(p['sigma_k']*(1 + 0.02*torch.randn((3,3,1), device=device)), 0, None),
            'weights' : p['weights']*(1+0.02*torch.randn((3,3), device = device))
        }
        
        self.update_params(p)


def perlin(shape:tuple, wavelengths:tuple, black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.

        Args:
        shape : (B,H,W) tuple or array, size of tensor
        wavelength : int, wavelength of the noise in pixels
        black_prop : percentage of black (0) regions. Defaut is .5=50% of black.
        device : device for returned tensor

        Returns :
        (B,3,H,W) torch tensor of perlin noise.
    """    
    B,H,W = tuple(shape)
    lams = tuple(int(wave) for wave in wavelengths)
    # Extend image so that its integer wavelengths of noise
    W_new=int(W+(lams[0]-W%lams[0]))
    H_new=int(H+(lams[1]-H%lams[1]))
    frequency = [H_new//lams[0],W_new//lams[1]]
    gen = torch.Generator(device=device) # for GPU acceleration
    gen.seed()
    # Strange 1/0.7053 factor to get images noise in range (-1,1), quirk of implementation I think...
    fp = FractalPerlin2D((B*3,H_new,W_new), [frequency], [1/0.7053], generator=gen)()[:,:H,:W].reshape(B,3,H,W) # (B*3,H,W) noise)

    return torch.clamp((fp+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1)
