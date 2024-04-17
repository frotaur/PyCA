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
            Converts _worldmap to a numpy array, and returns it in a pygame-plottable format (W,H,3).

            Can be overriden if you use another format for self._worldmap, instead of a torch (3,H,W) tensor.
        """
        return (255*self._worldmap.permute(2,1,0)).detach().cpu().numpy().astype(dtype=np.uint8)

