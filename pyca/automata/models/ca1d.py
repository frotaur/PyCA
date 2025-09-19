"""
    Variations on the 1D Cellular Automata.
"""

from ..automaton import Automaton
import colorsys
import random
import pygame
import torch



class ElementaryCA(Automaton):
    """
        1D Elementary Cellular Automata.
    """

    def __init__(self, size, wolfram_num : int, random: bool = False, device="cpu"):
        """
            Parameters:
            size : 2-uple (H,W)
                Shape of the CA world
            wolfram_num : int
                Number of the wolfram rule
            random : bool
                If True, the initial state of the automaton is random. Otherwise, the initial state is the middle cell set to 1.
        """
        super().__init__(size)
        # Below here I do what I need to do to initialize the automaton
        self.rule = self.convert_wolfram_num(wolfram_num) # (8,) tensor, rule[i] is 0 if the i'th neighborhood yields 0, 1 otherwise

        self.world = torch.zeros((self.w),dtype=torch.int) # Vector of W elements
        self.time = 0 # Current time step, to keep track for plotting the full evolution

        self.reset(random=random)
        
        self.random = random

        self.left_pressed=False
        self.right_pressed=False
        self.color = torch.tensor([1.,1.,1.]) # White color

    def process_event(self, event, camera=None):
        """
        CANC -> resets the automaton
        N -> pick a new random rule
        """
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                self.reset(random=self.random) 
            if(event.key == pygame.K_n): #Pressed the letter n
                # Picks a random rule
                rule = torch.randint(0,256,(1,)).item()
                self.change_num(rule)
                print('rule : ', rule)
            if(event.key == pygame.K_c):
                self.color = torch.rand((3,)) # Random color

        self._process_mouse_event(event,camera)


    def _process_mouse_event(self,event,camera):
        """
            Helper function, processes mouse events.
        """
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
                # Add interactions when dragging with left-click
        
            elif(self.right_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                # Add interactions when dragging with right-click

    def convert_wolfram_num(self,wolfram_num : int):
        """
            Converts a wolfram number to a rule tensor.
            A tensor, with 8 elements, 0 if the rule is 0, 1 if the rule is 1.
        """
        out = torch.zeros(8,dtype=torch.int8) # Prepare my arary of 8 binary elements
        for i in range(8):
            out[i] = (wolfram_num >> i) & 1
        
        # Now the array out contains the binary representation of wolfram_num

        return out.to(dtype=torch.int) # (Array of 8 elements, where out[i]=0 if the  neighborhood number i yields 0)
    
    def change_num(self,wolfram_num : int):
        """
            Changes the rule of the automaton to the one specified by wolfram_num
        """
        self.rule = self.convert_wolfram_num(wolfram_num)
    
    def reset(self, random = False):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))
        self.time=0

        if(random):
            self.world = torch.randint_like(self.world,0,2)
        else:
            self.world = torch.zeros((self.w),dtype=torch.int)
            self.world[self.w//2]=1
     
    def draw(self):
        # Draw should be called each step
        # We update the _worldmap tensor with the current state of the automaton
        # self._worldmap is a (3,H,W) tensor, which should be seen as an RGB image.
        # self.world is a (W,) array, but we want to set a (3,W) array, so we should duplicate the wrold.
        self._worldmap[:,self.time%self.h,:]=self.color[:,None]*self.world[None,:].float()
    
    def step(self):
        """
            Steps the automaton one timestep, recording the state of the world in self.world.
        """
        # One way to do it (which is not parallized ):

        # How to do it all at once
        
        indices = (torch.roll(self.world,shifts=(1))*4+self.world*2+torch.roll(self.world,shifts=(-1))*1) # (W,), compute in parallel all sums

        self.world=self.rule[indices] # This is the same as [ rule[indices[i]] for i in range(W)]

        self.time+=1


class TotalisticCA1D(Automaton):
    """
        General TOTALISTIC 1D Cellular Automaton, supports arbitrary neighborhoods and states.
        Totalistic means that the output state of a cell depends only on the sum of the states of the neighborhood.
    """

    def __init__(self, size, wolfram_num: int, r=1, k=2, random=False):
        """
            Parameters :
            size : 2-uple (H,W)
                Shape of the CA world
            wolfram_num : 0<=int<=k^(2r+1)-1, number of the wolfram rule 
            r : int, radius of the neighborhood
            k : int, number of states
            random : bool
                If True, the initial state of the automaton is random. Otherwise, the initial state is the middle cell set to a random non-zero value.
        """
        super().__init__(size)
        self.r = r
        self.k = k
        self.colors = self.get_color_list(k) # (k,3) tensor, contains the RGB values of the colors

        self.world = torch.zeros((self.w),dtype=torch.int)
        self.rule = self.convert_wolfram_num(wolfram_num) # (k^(2r+1),) tensor, rule[i] is the output state for a sum of i
        self.time = 0 # Current time step, to keep track for plotting the full evolution

        self.random = random

        self.reset(random=self.random)

    def process_event(self, event, camera=None):
        """
        DELETE -> resets the automaton
        N -> pick a new random rule
        UP -> increase the number of states
        DOWN -> decrease the number of states(resets the automaton)
        """
        if(event.type == pygame.KEYDOWN):
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.reset(random=self.random) 
                self.draw()
            if(event.key == pygame.K_n):
                # Picks a random rule
                rule = random.randint(0,self.k**((2*self.r+1)*(self.k-1)+1))
                self.change_num(rule)
                print('rule : ', rule)
            if event.key == pygame.K_UP:
                self.k = min(self.k+1, 4)
                rule = random.randint(0,self.k**((2*self.r+1)*(self.k-1)+1))
                self.change_num(rule)
                self.colors = self.get_color_list(self.k) # (k,3) tensor, contains the RGB values of the colors
            if event.key == pygame.K_DOWN:
                self.k = max(self.k-1, 2)
                rule = random.randint(0,self.k**((2*self.r+1)*(self.k-1)+1))
                self.change_num(rule)
                self.colors = self.get_color_list(self.k) # (k,3) tensor, contains the RGB values of the colors
                self.reset(random=self.random)  
    def get_color_list(self,n):
        colors = []
        zerohue = random.random()
        for i in range(n):
            hue = (i / n + zerohue)%1.  # Hue varies from 0 to 1, representing 0 to 360 degrees
            saturation = 0.7  # High saturation for vivid colors
            lightness = 0.5  # Balanced lightness for brightness
            if(i==0):
                lightness = 0.1
            # Convert HSL to RGB. colorsys returns values from 0 to 1
            rgb = torch.tensor(colorsys.hls_to_rgb(hue, lightness, saturation),dtype=torch.float) 
            colors.append(rgb)

        return torch.stack(colors) #(n,3) tensor, contains the RGB values of the colors


    def convert_wolfram_num(self,wolfram_num : int):
        """
            Converts a wolfram number to a rule tensor.

            Parameters :
            wolfram_num : 0<=int<=k^((2r+1))-1, number of the wolfram rule

            Returns :
            A tensor, with k^(2r+1) elements, containing the output state for each of the neighborhoods.
        """
        out = torch.zeros((2*self.r+1)*(self.k-1)+1,dtype=torch.int) # (2r+1,) tensor, rule[i] is the output state for a sum of i
        for i in range((2*self.r+1)*(self.k-1)+1):
            out[i] = wolfram_num//(self.k**i) % self.k # Extract the i'th digit of the wolfram number in base k
        
        return out.to(dtype=torch.int)    
    
    def change_num(self,wolfram_num : int):
        """
            Changes the rule of the automaton to the one specified by wolfram_num
        """
        self.rule = self.convert_wolfram_num(wolfram_num)
        self.colors = self.get_color_list(self.k)

    def draw(self):
        self._worldmap[:,self.time%self.h,:]=self.colors[self.world,:].permute(1,0) # (3,W), contains the RGB values of the colors
    
    def reset(self, random = False):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))
        self.time=0

        if(not random):
            self.world = torch.zeros((self.w),dtype=torch.int)
            self.world[self.w//2]=torch.randint(1,self.k,(1,)) # Random initial state
        else:
            self.world = torch.randint(0,self.k,(self.w,))
        
    
    def step(self):
        """
            Steps the automaton one timestep, recording the state of the world in self.world.
        """
        summed_states = sum([torch.roll(self.world,shifts=(i)) for i in range(-self.r,self.r+1)]) # (W,), compute in parallel all sums
        # print('k = ',self.k,' r = ',self.r,' summed_states = ',summed_states[self.w//2-1:self.w//2+2])
        self.world=self.rule[summed_states]

        self.time+=1