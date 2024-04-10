from ..Automaton import Automaton
import torch
from colorsys import hsv_to_rgb
import pygame
from torchvision.transforms import GaussianBlur


class LGCA(Automaton):
    """
        Simple FHP lattice gas cellular automaton, with support for boundaries
    """

    def __init__(self, size, device='cpu'):
        """
            Parameters:
            size : 2-uple (H,W)
                Shape of the CA world
        """
        super().__init__(size)

        self.world = torch.zeros((self.h,self.w,4),dtype=torch.bool,device=device) # 4 channels for left, right, up, down
        # Directions order convention : (North: 0, South: 1, Ouest: 2, East: 3)
        # Directions can be either 0 or 1 if there is/there isn't a particle in the cell
        self.walls = torch.zeros((self.h,self.w),dtype=torch.bool,device=device) # presence of walls

        self.palette = [[72,21,104],[35,138,141],[116,208,85],[253,231,37],[253,231,37]]
        for i in range(5):
            self.palette[i] = torch.tensor(self.palette[i],dtype=torch.float)/255.
        self.palette = torch.stack(self.palette,dim=0).to(device) # (5,3)

        self.device=device
        self.reset()

        self.vert_col_result = torch.tensor([False,False,True,True],dtype=torch.bool, device=device)
        self.horiz_col_result = torch.tensor([True,True,False,False],dtype=torch.bool, device=device)

        self.left_pressed=False
        self.right_pressed=False
        self.w_pressed=False
        self.brush_size = 5

        # For drawing : 
        self.Y, self.X = torch.meshgrid(torch.arange(0, self.h, device=self.device), torch.arange(0, self.w, device=self.device), indexing='ij')

    def process_event(self, event, camera=None):
        """
            Adds interactions : 
            - Left click and drag to add particles
            - Right click and drag to erase particles
            - Left click and drag while pressing W : add walls
            - Right click and drag while pressing W : erase walls
            - Scroll wheel to change brush size
            - Delete to reset the particles to homogeneous
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self.w_pressed = True
            if event.key == pygame.K_DELETE:
                self.reset()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                self.w_pressed = False
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
                set_mask = self.get_brush_slice(x,y)
                if(self.w_pressed):
                    # Add walls
                    self.walls[set_mask] = True
                    self.world[set_mask,:] = False
                else:
                    # Add particles
                    self.world[set_mask,:] = (self.world[set_mask,:] | 
                                        (torch.rand(self.world[set_mask,:].shape,device=self.device)<0.1)) & ~self.walls[set_mask][:,None]
            elif(self.right_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                set_mask = self.get_brush_slice(x,y)
                if(self.w_pressed):
                    # Erase walls
                    self.walls[set_mask] = False
                else:
                    # Erase particles
                    self.world[set_mask,:] = False
        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scroll wheel up
                self.brush_size += 1  # Increase brush size
            elif event.y < 0:  # Scroll wheel down
                self.brush_size -= 1  # Decrease brush size
    
            # Optional: Prevent brush size from going below 1
            self.brush_size = max(1, self.brush_size)
    
    def get_brush_slice(self, x, y):
        """Gets coordinate slices corresponding to the brush located at x,y"""
        set_mask = (self.Y-y)**2 + (self.X-x)**2 < self.brush_size**2
        return set_mask # (H,W)
    
    def reset(self):
        self.world = torch.rand((self.h,self.w,4),device=self.device)<0.3 # 20% filled
        # self.world[:,:,0] = torch.rand((self.h,self.w),device=self.device)<0.7 # 70% going north
        # Erase particles which are on walls, to avoid bugs
        self.world = torch.where(self.walls[:,:,None],False,self.world) # Hope the h,w,1 broadcasting works, if not, expand
        
    def step(self):
        """
            Steps the automaton one timestep.
        """

        # 1. Particle collision
        # Vertical
        horiz = self.world[:,:,0] & self.world[:,:,1] & ~(self.world[:,:,2] | self.world[:,:,3]) # N,S particles
        # assert (horiz & self.walls).any() == False, 'Horizontal collision on wall, bug somewhere'
        new_world = torch.where(horiz[:,:,None],self.vert_col_result[None,None,:],self.world) # Switch N,S to E,W where collided
        
        # Horizontal
        vert = self.world[:,:,2] & self.world[:,:,3] & ~(self.world[:,:,0] | self.world[:,:,1])
        assert (vert & horiz).any() == False, 'Double collision, bug somewhere'
        # assert (vert & self.walls).any() == False, 'Collision on horizontal wall, bug somewhere'
        new_world = torch.where(vert[:,:,None],self.horiz_col_result[None,None,:],new_world)

        self.world = new_world.clone()

        # 2. Wall collision
        for i in range(2):
            bounce = (self.world[:,:,2*i] | self.world[:,:,2*i+1]) & self.walls 
            # Exchange particles when bouncing
            new_world[:,:,2*i] = torch.where(bounce, self.world[:,:,2*i+1], self.world[:,:,2*i])
            new_world[:,:,2*i+1] = torch.where(bounce, self.world[:,:,2*i], self.world[:,:,2*i+1])
        self.world = new_world
        # 2. Propagation
        direc_dict = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        for i in range(4):
            self.world[:,:,i] = torch.roll(self.world[:,:,i],shifts=(direc_dict[i][0],direc_dict[i][1]),dims=(0,1))

    def draw(self):
        """
            Draws the LGCA world, hotter colors for denser regions
        """
        heat_sum = self.world.sum(dim=2).to(torch.int) # (H,W) tensor values between 0 and 4
        colors = self.palette[heat_sum] # (H,W,3) tensor
        self._worldmap = colors.permute(2,0,1) # (3,H,W) tensor
        self._worldmap = self._worldmap
        # Draw walls
        self._worldmap = torch.where(self.walls[None],torch.tensor([1.,0.,0.2],device=self.device)[:,None,None],self._worldmap) # Draw walls
        # self._worldmap = torch.where((self.walls & self.world.any(dim=-1))[None,:,:],torch.tensor([1.,1.,0.],device=self.device)[:,None,None],self._worldmap)