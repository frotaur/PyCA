import numpy as np
import torch

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

    def __init__(self, size, wolfram_num : int, init_state: torch.Tensor = None):
        """
            Parameters:
            size : 2-uple (H,W)
                Shape of the CA world
            wolfram_num : int
                Number of the wolfram rule
            init_state : torch.Tensor
                Initial state of the automaton. If None, it will be a single cell in the middle of the world.
        """
        super().__init__(size)
        self.rule = self.convert_wolfram_num(wolfram_num) # (8,) tensor, rule[i] is 0 if the i'th neighborhood yields 0, 1 otherwise

        self.world = torch.zeros((self.w),dtype=torch.int)

        if(init_state is not None):
            self.world = init_state
        else:
            self.world[self.w//2]=1
        
        self.time = 0 # Current time step, to keep track for plotting the full evolution


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
    
    def reset(self, init_state: torch.Tensor = None):
        """
            Resets the automaton to the initial state.
        """
        self._worlmap = torch.zeros((3,self.h,self.w))
        self.time=0
        self._worldmap = torch.zeros((3,self.h,self.w))
        self.time=0

        if(init_state is not None):
            self.world = init_state
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
