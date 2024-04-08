from ..Automaton import Automaton
import torch
import colorsys
from torchvision.transforms import Resize
import torch.nn.functional as F

class ReactionDiffusion(Automaton):
    """
        Reaction diffusion automaton.
    """ 


    def __init__(self, size, num_reagents=2, diffusion_coeffs:torch.Tensor = None, reaction_func=None,
                  dx=0.01, device='cpu'):
        """
            Parameters:
            size : tuple, size of the automaton
            num_reagents : int, number of reagents in the automaton
            diffusion_coeffs : float tensor, shape (num_reagents,), diffusion coefficients of the reagents
            reaction_func : reaction function. Given a tensor of shape (num_reagents,N) of floats in [0,1]
                should return a tensor of shape (num_reagents,N), the value of the reaction part. Here N
                is simply a 'batch' dimension, so we evaluate the function in parallel on N different states.
            dx : float, spatial discretization. dt automatically chosen to be dx^2/4
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
        self.dx = 1.
        self.dt = 1.


        min_dim = min(self.h,self.w)
        # Normalized size of box so that smaller dim is [0,1]
        self.h_len = self.h/min_dim
        self.w_len = self.w/min_dim


        self.grid = torch.stack(torch.meshgrid(torch.arange(0,self.h_len,dx),torch.arange(0,self.w_len,dx),indexing='ij'),dim=-1).to(device) # (H,W,2), grid[x,y] = (x,y)
        self.Nh, self.Nw = self.grid.shape[0], self.grid.shape[1]

        self.u = torch.rand((self.Nh,self.Nw,num_reagents),dtype=torch.float, device=device) # (N_h,N_w,num_reagents), concentration of reagents
        x = torch.linspace(-1, 1, steps=self.Nw).unsqueeze(0).repeat(self.Nh, 1)
        y = torch.linspace(-1, 1, steps=self.Nh).unsqueeze(1).repeat(1, self.Nw)
        self.u[:,:,1] = (x**2+y**2<0.001).to(torch.float)
        self.u[:,:,0] = torch.where(self.u[:,:,1]>0,0.,1.)
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

        self.r_colors = [torch.ones(3,dtype=torch.float)]
        for _ in range(num_reagents-1):
            hue = torch.rand(1).item()
            print('#hue : ', hue)
            value = 0.5*torch.rand(1).item()+0.5
            saturation = 0.5*torch.rand(1).item()+0.5
            self.r_colors.append(torch.tensor(colorsys.hsv_to_rgb(hue, saturation, value),dtype=torch.float))
            

        self.r_colors = torch.stack(self.r_colors,dim=0).to(device) # (num_reagents,3)
        print('colors : ', self.r_colors)
        self.resizer = Resize((self.h,self.w),antialias=True)

        self.lapl_kern =torch.tensor([
                        [0.05, 0.2, 0.05],
                        [0.2, -1.0, 0.2],
                        [0.05, 0.2, 0.05]
                    ]).unsqueeze(0).unsqueeze(0).expand(self.num_reagents,-1,-1,-1).to(device) # (1,1,3,3)

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

    @torch.no_grad()
    def step(self):
        """
            Steps the automaton one timestep.
        """
        self.u = self.u+self.dt*(self.D*self.lapl(self.u)+ self.compute_R(self.u))
        # self.u = self.u+self.dt*self.lapl(self.u)
    
    def draw(self):
        """
            Draws a representation of the worldmap, reshaping in case the dx resolution is too high/low.
        """

        # u : (N_h,N_w,num_reagents,1), r_colors : (num_reagents,3)
        new_world = ((self.u[...,None]) * self.r_colors[None,None]).mean(dim=2)/(self.u[...,None].mean(dim=2)+1e-5) # (H,W,3)
        self._worldmap = self.resizer(new_world.permute(2,0,1)) # (3,H,W) resized to match window
