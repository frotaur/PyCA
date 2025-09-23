from ..automaton import Automaton
import torch
import colorsys
from torchvision.transforms import Resize
import torch.nn.functional as F
import pygame


class ReactionDiffusion(Automaton):
    """
        Reaction diffusion automaton.
    """ 

    def __init__(self, size, num_reagents=2, diffusion_coeffs:torch.Tensor = None, reaction_func=None,
                  dx=None, device='cpu'):
        """
            Parameters:
            size : tuple, size of the automaton
            num_reagents : int, number of reagents in the automaton
            diffusion_coeffs : float tensor, shape (num_reagents,), diffusion coefficients of the reagents
            reaction_func : reaction function. Given a tensor of shape (num_reagents,N) of floats in [0,1]
                should return a tensor of shape (num_reagents,N), the value of the reaction part. Here N
                is simply a 'batch' dimension, so we evaluate the function in parallel on N different positions.
            dx : float or None, spatial discretization. If None, dx=1 and window is not scaled to [0,1].
                If not None, scales window to [0,1], and uses set dx. In both cases, dt automatically chosen to be dx^2/4
            device : str, device to use

            ###################################################################
            reaction_func example, analytic case :
            def reac_ex(u):
                return torch.stack([u[:,0]-3*u[:,1],u[:,1]-u[:,0]],dim=0)
            
            For now, this works only for 'analytic' reaction functions.
            ###################################################################
            In the future, I would like to add support for numerical functions, 
            i.e. if you have a point cloud that represent the reaction function,
            you can still use it.

            One idea is to have the discretized function as a tensor f of shape (num_reagents, N, N, ..., N),
            where N is repeated 'num_reagents' times. Here N*Dx=1, i.e. N is the number of points in the 
            regular discretization. f[:,i,j,...,k] is the value of the function at the point (i*Dx,j*Dx,...,k*Dx).

            If you have this tensor, then given u of shape (num_reagents,) you compute ceil(u/Dx) and floor(u/Dx),
            and then interpolate the function f at these points. E.g. if u/Dx is integer, then the value of the function
            is f[:,u/Dx]. Should work also with batched inputs !
            ###################################################################
        """
        super().__init__(size)

        self.num_reagents = num_reagents
        self.reaction_func = reaction_func
        self.device=device

        if(dx is None):
            self.dx = 1.
            # No scaling, 1 pixel = 1 unit
            self.h_len = self.h
            self.w_len = self.w
            self.resizer=None
        else:
            self.dx = dx
            # Scale window to [0,1]
            min_dim = min(self.h,self.w)
            # Normalized size of box so that smaller dim is [0,1]
            self.h_len = self.h/min_dim
            self.w_len = self.w/min_dim
            self.resizer = Resize((self.h,self.w),antialias=False)


        self.dt = self.dx**2*0.6 # Use diffusion coeffs max 0.25 for stability



        self.grid = torch.stack(torch.meshgrid(torch.arange(0,self.h_len,self.dx),torch.arange(0,self.w_len,self.dx),indexing='ij'),dim=-1).to(device) # (H,W,2), grid[x,y] = (x,y)
        self.Nh, self.Nw = self.grid.shape[0], self.grid.shape[1]
        self.reset()


        if(reaction_func is None):
            # Default reaction function
            self.R = lambda u : torch.zeros_like(u) # Pure diffusion if no reaction function
        else :
            self.R = reaction_func

        test = torch.randn((self.num_reagents,30),dtype=torch.float,device=device)
        out = self.R(test)
        assert out.shape == test.shape, 'Reaction function problem; expected shape {}, got shape {}'.format(test.shape,out.shape)

        if(diffusion_coeffs is None):
            self.D = (1.)*torch.ones((num_reagents),dtype=torch.float,device=device)[:,None,None] # (num_reagents,1,1)
            self.D[1,:,:] = .5
        else:
            assert diffusion_coeffs.shape == (num_reagents,), 'Diffusion coefficients shape should be (num_reagents,), got shape {}'.format(diffusion_coeffs.shape)
            self.D = diffusion_coeffs[:,None,None].to(device) # (num_reagents,1,1)


        self.recolor()


        self.lapl_kern =torch.tensor([
                        [0.2, 0.8, 0.2],
                        [0.8, -4.0, 0.8],
                        [0.2, 0.8, 0.2]
                    ])[None,None].expand(self.num_reagents,-1,-1,-1).to(device) # (2,1,3,3)

        self.brush_size = 10
        self.left_pressed=False
        self.right_pressed=False
        self.selected_reagent = 0
        self.reagent_mask = torch.zeros((self.num_reagents),dtype=torch.bool,device=device)
        self.reagent_mask[self.selected_reagent] = True

        self.stepnum=1
        self.renormalize = None
    def reset(self):
        self.u = torch.zeros((self.num_reagents,self.Nh,self.Nw),dtype=torch.float, device=self.device) # (num_reagents,N_h,N_w), concentration of reagents
        x = torch.linspace(-self.w/2, self.w/2, steps=self.Nw).unsqueeze(0).repeat(self.Nh, 1)
        y = torch.linspace(-self.h/2, self.h/2, steps=self.Nh).unsqueeze(1).repeat(1, self.Nw)
        self.u[1,:,:] = (x**2+y**2<30).to(torch.float)
        self.u[0,:,:] = torch.where(self.u[1,:,:]>0,0.,1.)


    def recolor(self):
        self.r_colors = [torch.ones(3,dtype=torch.float)]
        for _ in range(self.num_reagents-1):
            hue = torch.rand(1).item()
            value = 0.2*torch.rand(1).item()+0.8
            saturation = 0.2*torch.rand(1).item()+0.8
            self.r_colors.append(torch.tensor(colorsys.hsv_to_rgb(hue, saturation, value),dtype=torch.float))
            

        self.r_colors = torch.stack(self.r_colors,dim=0).to(self.device) # (num_reagents,3)
        print('self.r_colors',self.r_colors)    


    def lapl(self, u):
        """
            Laplacian of u, computed with finite differences.

            Parameters:
            u : tensor, shape (num_reagents,H,W), concentration of reagents

            Returns:
            tensor, shape (num_reagents,H,W), laplacian of u
        """

        # u = F.pad(u,(1,1,1,1),'circular')
        lapl = torch.clamp(F.conv2d(u[None],self.lapl_kern,groups=self.num_reagents,padding=1).squeeze(0),min=-5,max=5)

        return lapl # (num_reagents,H,W)
    
    def compute_R(self,u):
        u = u.reshape(self.num_reagents,self.Nh*self.Nw) # (num_reagents,H*W)
        u = self.R(u) # (num_reagents,H*W) reactions
        u = u.reshape(self.num_reagents,self.Nh,self.Nw) # (H,W,num_reagents) reaction

        return u
    
    def process_event(self, event, camera=None):
        """
            - Left click and drag to add selected chemical
            - Right click and drag to remove selected chemical
            - Scroll wheel to change chemical
            - Delete to reset the state to homogeneous
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.reset()
            if event.key == pygame.K_c:
                self.recolor()
            if event.key == pygame.K_1:
                # If ctrl is pressed, decrease Da
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.D[self.selected_reagent] = torch.clip(self.D[self.selected_reagent]-0.1,0.1,3.)
                else :
                    self.D[self.selected_reagent] = torch.clip(self.D[self.selected_reagent]+0.1,0.1,3.)
                print('D{} : '.format(self.selected_reagent),self.D[self.selected_reagent])
    
        mouse = self.get_mouse_state(camera)

        if event.type == pygame.MOUSEMOTION:
            if(mouse.left):
                set_mask = self.get_brush_slice(mouse.x,mouse.y)[None] # (1,H,W)
                set_mask = set_mask & self.reagent_mask[:,None,None]
                # Add particles
                self.u[set_mask] = .8
            elif(mouse.right):
                set_mask = self.get_brush_slice(mouse.x,mouse.y)[None] # (1,H,W)
                set_mask = set_mask & self.reagent_mask[:,None,None]
                # Add particles
                self.u[set_mask] = 0.

        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scroll wheel up
                self.selected_reagent = (self.selected_reagent+1)%self.num_reagents  # Increase brush size
                self.reagent_mask = torch.zeros((self.num_reagents),dtype=torch.bool,device=self.u.device)
                self.reagent_mask[self.selected_reagent] = True
            elif event.y < 0:  # Scroll wheel down
                self.selected_reagent = (self.selected_reagent-1)%self.num_reagents  # Increase brush size
                self.reagent_mask = torch.zeros((self.num_reagents),dtype=torch.bool,device=self.u.device)
                self.reagent_mask[self.selected_reagent] = True
    
    def get_brush_slice(self, x, y):
        """Gets coordinate slices corresponding to the brush located at x,y"""
        set_mask = (self.grid[:,:,0]-y)**2 + (self.grid[:,:,1]-x)**2 < self.brush_size**2
        return set_mask # (H,W)
    
    @torch.no_grad()
    def step(self):
        """
            Steps the automaton one timestep.
        """
        for _ in range(self.stepnum): # many steps per step, because too slow apparently
            self.u = self.u+(self.D*self.lapl(self.u)+ self.compute_R(self.u))*self.dt


    def draw(self):
        """
            Draws a representation of the worldmap, reshaping in case the dx resolution is too high/low.
        """

        # u : (num_reagents,N_h,N_w,1), r_colors : (num_reagents,1,1,3)
        if(self.renormalize is not None):
            self.renormalize = torch.maximum(self.u.reshape(self.num_reagents,-1).max(dim=1)[0],self.renormalize) # (num_reagents,)
            drawu = self.u[...,None]/self.renormalize[:,None,None,None]
        else :
            drawu = self.u[...,None]
        new_world = ((drawu) * self.r_colors[:,None,None]).sum(dim=0)/(drawu.sum(dim=0)+1e-6)*drawu.max(dim=0)[0] # (H,W,3)

        select_size= 6
        new_world[:select_size,self.w-select_size:,:] = self.r_colors[self.selected_reagent] # Draw a square of the selected reagent in the top left corner
        # Put a black border around the square
        new_world[select_size,self.w-select_size:,:] = 0
        new_world[:select_size+1,self.w-select_size,:] = 0

        if(self.resizer is not None):
            self._worldmap = self.resizer(new_world.permute(2,0,1)) # (3,H,W) resized to match window
        else:
            self._worldmap = new_world.permute(2,0,1)


class GrayScott(ReactionDiffusion):
    """
        Reaction Diffusion Gray-Scott model. Selected chemical color displayed on the top left corner.
    """

    def __init__(self, size, Da=1.,Db=0.5,f=.06100,k=.06264,device='cpu'):
        """
            Parameters:
            size : tuple, size of the automaton
            device : str, device to use
        """
        self.f = f
        self.k = k
        ## Adjust diffusion because of their shitty laplacian
        Da = Da/4.
        Db = Db/4.

        self.stepnum=20
        
        def gray_scott_reaction(u):
            return torch.stack([-u[0]*u[1]**2+self.f*(1-u[0]),u[0]*u[1]**2-(self.k+self.f)*u[1]],dim=0) # (2, N)
        
        super().__init__(size, num_reagents=2, diffusion_coeffs=torch.tensor([Da,Db]), reaction_func=gray_scott_reaction, device=device)
        self.renormalize = None
        print('dt is : ', self.dt)


    def process_event(self, event, camera=None):
        """
            - Left click and drag to add selected chemical
            - Right click and drag to remove selected chemical
            - Scroll wheel to change chemical
            - Delete to reset the state to homogeneous
        """
        super().process_event(event, camera)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                # If ctrl is pressed, decrease Da
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.D[0] = torch.clip(self.D[0]-0.1,0.1,3.)
                else :
                    self.D[0] = torch.clip(self.D[0]+0.1,0.1,3.)
            if event.key == pygame.K_2:
                # If ctrl is pressed, decrease Db
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.D[1] = torch.clip(self.D[1]-0.1,0.1,3.)
                else :
                    self.D[1] = torch.clip(self.D[1]+0.1,0.1,3.)
            if event.key == pygame.K_f:
                # If ctrl is pressed, decrease f
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.f = max(self.f-0.001,0.01)
                else :
                    self.f = self.f+0.001
                print('f : ',self.f)
            if event.key == pygame.K_k:
                # If ctrl is pressed, decrease k
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.k = max(self.k-0.001,0.01)
                else :
                    self.k = self.k+0.001
                print('k : ',self.k)



class BelousovZhabotinsky(ReactionDiffusion):
    """
        Belousov-Zhabotinsky model.
    """

    def __init__(self, size, Da=0.4,Db=.4,Dc=.4,alpha=.3,beta=.2,gamma=.2,device='cpu'):
        """
            Parameters:
            size : tuple, size of the automaton
            device : str, device to use
        """
        Da,Db,Dc = Da/5.,Db/5.,Dc/5.

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.renormalize = None
        def belousov_zhabotinsky_reaction(u):
            return torch.stack([u[0]*(self.alpha*u[1]-self.gamma*u[2]),u[1]*(self.beta*u[2]-self.alpha*u[0]),u[2]*(self.gamma*u[0]-self.beta*u[1])],dim=0) # (3, N)
        
        super().__init__(size, num_reagents=3, diffusion_coeffs=torch.tensor([Da,Db,Dc]), reaction_func=belousov_zhabotinsky_reaction, device=device)
    
        print('dt is : ', self.dt)
        self.u = torch.zeros_like(self.u)
        self.u[0] =1.
        self.u[1:] = self.u[1:] + 0.01*torch.rand_like(self.u[1:])
        self.stepnum=1

        self.renormalize = torch.ones((3),dtype=torch.float,device=device) # (num_reagents,)

    def step(self):
        """
            Modified Reaction Diffusion step to add a clipping to prevent runaway
        """
        super().step()
        self.u = torch.clamp(self.u,0.,1.5)

    def process_event(self, event, camera=None):
        """
            - Left click and drag to add selected chemical
            - Right click and drag to remove selected chemical
            - Scroll wheel to change chemical
            - Delete to reset the state to homogeneous
        """
        super().process_event(event, camera)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a :
                # If ctrl is pressed, decrease alpha
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.alpha = max(self.alpha-0.01,0.01)
                else :
                    self.alpha = self.alpha+0.01
                print('alpha : ',self.alpha)
            if event.key == pygame.K_b :
                # If ctrl is pressed, decrease beta
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.beta = max(self.beta-0.01,0.01)
                else :
                    self.beta = self.beta+0.01
                print('beta : ',self.beta)
            if event.key == pygame.K_g :
                # If ctrl is pressed, decrease gamma
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.gamma = max(self.gamma-0.01,0.01)
                else :
                    self.gamma = self.gamma+0.01
                print('gamma : ',self.gamma)

class Brusselator(ReactionDiffusion):
    """
        Brusselator model.
    """
    def __init__(self, size, Da=.64,Db=.4,alpha=0.7,beta=2.5,device='cpu'):
        """
            Parameters:
            size : tuple, size of the automaton
            device : str, device to use
        """
        Da,Db = Da,Db

        self.alpha = alpha
        self.beta = beta
        self.gamma = 0.2
        def brusselator_reaction(u):
            return torch.stack([self.gamma*(u[0]**2*u[1]-self.beta*u[0]-u[0]+self.alpha),self.gamma*(self.beta*u[0]-u[0]**2*u[1])],dim=0) # (2, N)
        
        super().__init__(size, num_reagents=2, diffusion_coeffs=torch.tensor([Da,Db]), reaction_func=brusselator_reaction, device=device)
        
        print('dt is : ', self.dt)
        self.u = torch.zeros_like(self.u)
        self.u[1] =.1
        self.stepnum=1
        self.dt=0.4
        self.brush_size=10

        self.renormalize = torch.ones((2),dtype=torch.float,device=device) # (num_reagents,)

    def process_event(self, event, camera=None):
        """
            - Left click and drag to add selected chemical
            - Right click and drag to remove selected chemical
            - Scroll wheel to change chemical
            - Delete to reset the state to homogeneous
        """
        super().process_event(event, camera)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a :
                # If ctrl is pressed, decrease alpha
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.alpha = max(self.alpha-0.1,0.1)
                else :
                    self.alpha = self.alpha+0.1
                print('alpha : ',self.alpha)
            if event.key == pygame.K_b :
                # If ctrl is pressed, decrease beta
                if(pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.beta = max(self.beta-0.1,0.1)
                else :
                    self.beta = self.beta+0.1
                print('beta : ',self.beta)