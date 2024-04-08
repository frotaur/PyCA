from ..Automaton import Automaton
import torch
from colorsys import hsv_to_rgb
import pygame

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

        self.brush_size = 5
        # x,y=25,25
        # slices = slice(max(int(y)-5,0),min(int(y)+5,self.h-1)), slice(max(int(x)-5,0),min(int(x)+5,self.w))
        # self.walls[slices[0],slices[1]] = True
        # self.world[slices[0],slices[1],:] = False

    def process_event(self, event, camera=None):
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
                slices = slice(max(int(y)-1,0),min(int(y)+self.brush_size,self.h-self.brush_size)), slice(max(int(x)-self.brush_size,0),min(int(x)+self.brush_size,self.w))
                self.walls[slices[0],slices[1]] = True
                self.world[slices[0],slices[1],:] = False
            elif(self.right_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                # Add interactions when dragging with right-click
                self.walls[int(y),int(x)] = False

    def reset(self):
        self.world = torch.rand((self.h,self.w,4),device=self.device)<0.2 # 20% filled
        self.world[:,:,0] = torch.rand((self.h,self.w),device=self.device)<0.7 # 70% going north
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

        # Draw walls
        self._worldmap = torch.where(self.walls[None],torch.tensor([1.,0.,0.2],device=self.device)[:,None,None],self._worldmap) # Draw walls
        self._worldmap = torch.where((self.walls & self.world.any(dim=-1))[None,:,:],torch.tensor([1.,1.,0.],device=self.device)[:,None,None],self._worldmap)