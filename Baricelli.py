"""
    Baricelli automata, both 1D and 2D.
"""
from Automaton import Automaton
import torch
from matplotlib.colors import hsv_to_rgb
import pygame


class Baricelli1D(Automaton):
    """
        Baricelli model in 1D. 
    """

    def __init__(self, size, n_species : int=6, reprod_collision=False):
        """
            Params:
            size : 2-uple (H,W)
                Size of the spacetime
            n_species : int
                Allowed integers will be -n_species to n_species
            reprod_collision : bool
                If False, will not attempt to reproduce on an already lit pixel. 
                If True, attempting to reproduce on an already lit pixel will result in a collision and annihilation.
                In BOTH cases, two attempt at reproduction on the same cell result in collisions.
        """
        super().__init__(size)

        self.time=0
        self.speciesnum = n_species
        self.repcol = reprod_collision

        self.positions = torch.arange(self.w) # (W,), to not have to make it each time
        self.world = torch.randint(-n_species,n_species+1,(self.w,),dtype=torch.int)

    
    def step(self):
        """
            Makes one step of the automaton
        """
        new_world = torch.zeros_like(self.world) # (W,), copy of the world

        ## First, move
        tar_pos = ((self.positions+self.world)%self.w)[self.world!=0] # (N,), target positions of nonzero cells
        movers = self.world[self.world!=0] # (N,), species of movers
        move_mask, move_locs = self._move_collision(tar_pos) 

        new_world[move_locs] = movers[move_mask] # (W,), update the world with successful movers, others are dead

        ## Then reproduction
        # No repcol : no reproduction on already lit cells
        reprod_attempt = (self.world!=0) & (new_world!=0) # (W,), mask of cells that can attempt reproduction
        reprod_parent = new_world[reprod_attempt] # (N,), parent species

        rep_pos= (self.positions[reprod_attempt]+self.world[reprod_attempt]-new_world[reprod_attempt])%self.w # (N,), positions of reproduction

        rep_mask, rep_locs = self._move_collision(rep_pos) 

        reprod_parent = reprod_parent[rep_mask] # (N',), parent species of non-colliding reproducers

        reprod_success =(new_world[rep_locs]==0) | (new_world[rep_locs]==reprod_parent) # (N',) Success mask, if already empty or same species
        new_world[rep_locs[reprod_success]] = reprod_parent[reprod_success] # (W,), update the world with successful reproducers        
        
        self.time+=1

        self.world = new_world


    def _move_collision(self, tar_pos):
        """
            Compute which particules are allowed to move/reproduce, according the possible tar_positions
            Params :
            tar_pos : (N,), tensor of the target positions of the particles

            returns : (N',), (N',) mask of successfull movers, and their target positions
        """
        can_move = torch.bincount(tar_pos,minlength=self.w)==1 # (W,) mask,  can_move[i] True is only one particle wants to go to position i

        success_mask = (can_move)[tar_pos] # (N',), mask of successful movers. success_mask[i] is True if mover number i (ordered) can move
        success_moves = tar_pos[success_mask] # (N',), indices of arrival of successful movers success_moves[i] is the position of mover i

        return success_mask, success_moves

    def get_color_world(self):
        """
            Return colorized sliced world

            Returns : (W,3) tensor of floats
        """
        colorize=torch.zeros((self.w,3),dtype=torch.float) # (W,3)

        colorize[:,0]=self.world/(self.speciesnum*2.)+.5
        colorize[:,1]=.7
        colorize[:,2]=.8 * torch.where(self.world==0,0,1)
        colorize=torch.tensor(hsv_to_rgb(colorize.numpy())) # (W,3)
        
        return colorize
    
    def reset(self):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))
        self.time=0

        self.world = torch.randint(-self.speciesnum,self.speciesnum+1,(self.w,),dtype=torch.int)
        # self.world = torch.zeros_like(self.world)
        # self.world[self.w//2-5]=1
        # self.world[self.w//2+2]=-2

    def process_event(self, event, camera=None):
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                # ONLY WORKS WITH CA1D ! REMOVE/add reset method to use with other automata
                self.reset() 
                self.draw()
            if(event.key == pygame.K_DOWN):
                self.step()

    def draw(self):
        """
            Draws the current state of the automaton, using arbitrary coloration which looks nice.
        """
        self._worldmap[:,self.time%self.h,:]=self.get_color_world().permute(1,0) # (3,W)