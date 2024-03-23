import numpy as np
import torch
import pygame

class Automaton :
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention, the world tensor has shape
        (3,H,W). It contains float values between 0 and 1, which
        are mapped to 0 255 when returning output, and describes how the
        world is 'seen' by an observer.

    """

    def __init__(self,size):
        """     
        Parameters :
        size : 2-uple (H,W)
            Shape of the CA world
        """
        self.h, self.w  = size
        self.size= size

        self._worldmap = torch.zeros((3,self.h,self.w)) # (3,H,W), contains a 2D 'view' of the CA world
    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    def draw(self):
        """
            This function should update the self._worldmap tensor
        """
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')
    
    def process_event(self,event,camera=None):
        """
            Processes a pygame event, if needed.

            Parameters:
            event : pygame.event
                The event to process
            camera : Camera
                The camera object. Might be needed to convert mouse positions to world coordinates.
                Use camera.convert_mouse_pos(pygame.mouse.get_pos()) to convert the mouse position to world coordinates.
        """
        pass

    @property
    def worldmap(self):
        """
            Converts _worldmap to a numpy array, and returns it in a pygame-plottable format (H,W,3).

            Can be overriden if you use another format for self._worldmap, instead of a torch (3,H,W) tensor.
        """
        return (255*self._worldmap.permute(2,1,0)).detach().numpy().astype(dtype=np.uint8)



class CA1D(Automaton):
    """
        Class that implements all 1D Cellular Automata, the "Elementary" cellular automata.
    """

    def __init__(self, size, wolfram_num : int, random: bool = False):
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
        self.rule = self.convert_wolfram_num(wolfram_num) # (8,) tensor, rule[i] is 0 if the i'th neighborhood yields 0, 1 otherwise

        self.world = torch.zeros((self.w),dtype=torch.int)
        self.time = 0 # Current time step, to keep track for plotting the full evolution

        self.reset(random=random)
        
        self.random = random

        self.left_pressed=False
        self.right_pressed=False

    def process_event(self, event, camera=None):
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                self.reset(random=self.random) 
            if(event.key == pygame.K_n):
                # Picks a random rule
                rule = torch.randint(0,256,(1,)).item()
                self.change_num(rule)
                print('rule : ', rule)

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
        out = torch.zeros(8,dtype=torch.int8)
        for i in range(8):
            out[i] = (wolfram_num >> i) & 1
        
        return out.to(dtype=torch.int)
    
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
        self._worldmap[:,self.time%self.h,:]=self.world[None,:].float()
    
    def step(self):
        """
            Steps the automaton one timestep, recording the state of the world in self.world.
        """
        indices = (torch.roll(self.world,shifts=(1))*4+self.world*2+torch.roll(self.world,shifts=(-1))) # (W,), compute in parallel all sums

        self.world=self.rule[indices]

        self.time+=1
