from ..Automaton import Automaton
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
                return torch.stack([u[0]-3*u[1],u[1]-u[0]],dim=0)
            
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


        self.dt = self.dx**2/4

        self.step_increment = 0.25 # This is dt/dx^2




        self.grid = torch.stack(torch.meshgrid(torch.arange(0,self.h_len,self.dx),torch.arange(0,self.w_len,self.dx),indexing='ij'),dim=-1).to(device) # (H,W,2), grid[x,y] = (x,y)
        self.Nh, self.Nw = self.grid.shape[0], self.grid.shape[1]

        self.u = torch.zeros((self.Nh,self.Nw,num_reagents),dtype=torch.float, device=device) # (N_h,N_w,num_reagents), concentration of reagents
        x = torch.linspace(-self.w/2, self.w/2, steps=self.Nw).unsqueeze(0).repeat(self.Nh, 1)
        y = torch.linspace(-self.h/2, self.h/2, steps=self.Nh).unsqueeze(1).repeat(1, self.Nw)
        self.u[:,:,0] = (x**2+y**2<100).to(torch.float)
        # self.u[:,:,0] = torch.where(self.u[:,:,1]>0,0.,1.)
        self.u = self.u.to(device)

        if(reaction_func is None):
            # Default reaction function
            # Override num_reagents
            self.num_reagents = 2
            f=.0545
            k=.062
            self.R = lambda u : torch.stack([-u[0]*u[1]**2+f*(1-u[0]),u[0]*u[1]**2-(k+f)*u[1]],dim=0) # Change this to nice one
        else :
            self.R = reaction_func

        test = torch.randn((self.num_reagents,30))
        out = self.R(test)
        assert out.shape == test.shape, 'Reaction function problem; expected shape {}, got shape {}'.format(test.shape,out.shape)

        if(diffusion_coeffs is None):
            self.D = (1.)*torch.ones((num_reagents),dtype=torch.float,device=device)[None,None,:] # (1,1,num_reagents)
            self.D[:,:,1] = .2
        else:
            assert diffusion_coeffs.shape == (num_reagents,), 'Diffusion coefficients shape should be (num_reagents,), got shape {}'.format(diffusion_coeffs.shape)
            self.D = diffusion_coeffs[None,None,:] # (1,1,num_reagents)


        self.recolor()


        self.lapl_kern =torch.tensor([
                        [0.2, 0.8, 0.2],
                        [0.8, -4.0, 0.8],
                        [0.2, 0.8, 0.2]
                    ]).unsqueeze(0).unsqueeze(0).expand(self.num_reagents,-1,-1,-1).to(device) # (1,1,3,3)

        self.brush_size = 10
        self.left_pressed=False
        self.right_pressed=False
        self.selected_reagent = 0
        self.reagent_mask = torch.zeros((self.num_reagents),dtype=torch.bool,device=device)
        self.reagent_mask[self.selected_reagent] = True

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
            u : tensor, shape (H,W,num_reagents), concentration of reagents

            Returns:
            tensor, shape (H,W,num_reagents), laplacian of u
        """

        # u_xplus = torch.roll(u,shifts=(-1,0),dims=(0,1))
        # u_xminus = torch.roll(u,shifts=(1,0),dims=(0,1))
        # u_yplus = torch.roll(u,shifts=(0,-1),dims=(0,1))
        # u_yminus = torch.roll(u,shifts=(0,1),dims=(0,1))

        # lapl = (u_xplus+u_xminus+u_yplus+u_yminus-4*u) # (H,W,num_reagents)
        u = F.pad(u.permute(2,0,1),(1,1,1,1),'circular')
        lapl = F.conv2d(u.unsqueeze(0),self.lapl_kern,groups=self.num_reagents).squeeze(0).permute(1,2,0)
        return lapl # (H,W,num_reagents)
    
    def compute_R(self,u):
        u = u.permute(2,0,1).reshape(self.num_reagents,self.Nh*self.Nw) # (num_reagents,H*W)
        u = self.R(u) # (num_reagents,H*W) reactions
        u = u.reshape(self.num_reagents,self.Nh,self.Nw).permute(1,2,0) # (H,W,num_reagents) reaction

        return u
    
    def process_event(self, event, camera=None):
        """
            Adds interactions : 
            - Left click and drag to add selected chemical
            - Right click and drag to remove selected chemical
            - Scroll wheel to change chemical
            - Delete to reset the state to homogeneous
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:
                self.reset()
            if event.key == pygame.K_c:
                self.recolor()
        if event.type == pygame.MOUSEBUTTONDOWN :
            if(event.button == 1):
                self.left_pressed=True
            if(event.button ==3):
                self.right_pressed=True
        if event.type == pygame.MOUSEBUTTONUP:
            if(event.button==1):
                self.left_pressed=False
            elif(event.button==3):
                self.right_pressed=False
    
        if event.type == pygame.MOUSEMOTION:
            if(self.left_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                set_mask = self.get_brush_slice(x,y)[...,None] # (H,W,1)
                set_mask = set_mask & self.reagent_mask[None,None,:]
                # Add particles
                self.u[set_mask] = 1.
            elif(self.right_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                set_mask = self.get_brush_slice(x,y)[...,None] # (H,W,1)
                set_mask = set_mask & self.reagent_mask[None,None,:]
                # Remove particles
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
        self.u = self.u+self.step_increment*self.D*self.lapl(self.u)#+ self.dt*self.compute_R(self.u)
        # self.u = self.u+self.dt*self.lapl(self.u)
    
    def draw(self):
        """
            Draws a representation of the worldmap, reshaping in case the dx resolution is too high/low.
        """

        # u : (N_h,N_w,num_reagents,1), r_colors : (1,1,num_reagents,3)
        new_world = ((self.u[...,None]) * self.r_colors[None,None]).sum(dim=2)/(self.u[...,None].sum(dim=2)+1e-6)*self.u[...,None].max(dim=2)[0] # (H,W,3)

        select_size= 4
        new_world[:select_size,:select_size,:] = self.r_colors[self.selected_reagent] # Draw a square of the selected reagent in the top left corner
        # Put a black border around the square
        new_world[select_size,:select_size+1,:] = 0
        new_world[:select_size+1,select_size,:] = 0

        if(self.resizer is not None):
            self._worldmap = self.resizer(new_world.permute(2,0,1)) # (3,H,W) resized to match window
        else:
            self._worldmap = new_world.permute(2,0,1)


class GrayScott(ReactionDiffusion):
    """
        Gray-Scott model.
    """

    def __init__(self, size, Da=1.,Db=0.5,f=0.055,k=.062,device='cpu'):
        """
            Parameters:
            size : tuple, size of the automaton
            device : str, device to use
        """
        def gray_scott_reaction(u):
            return torch.stack([-u[0]*u[1]**2+f*(1-u[0]),u[0]*u[1]**2-(k+f)*u[1]],dim=0) # (2,N)
        
        super().__init__(size, num_reagents=2, diffusion_coeffs=torch.tensor([Da,Db]), reaction_func=gray_scott_reaction, device=device)

        