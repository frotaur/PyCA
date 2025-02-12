import numpy as np
import torch
from textwrap import dedent
import pygame
from easydict import EasyDict


class Automaton:
    """
    Class that internalizes the rules and evolution of
    an Alife model. By default, the world tensor has shape
    (3,H,W) and should contain floats with values in [0.,1.].
    """

    def __init__(self, size):
        """
        Parameters :
        size : 2-uple (H,W)
            Shape of the CA world
        """
        self.h, self.w = size
        self.size = size

        self._worldmap = torch.zeros((3, self.h, self.w), dtype=float)  # (3,H,W), contains a 2D 'view' of the CA world

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        """
        This method should be overriden. It should update the self._worldmap tensor,
        drawing the current state of the CA world. self._worldmap is a torch tensor of shape (3,H,W).
        If you choose to use another format, you should override the worldmap property as well.
        """
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')
    
    def process_event(self, event, camera=None):
        """
        Processes a pygame event, if needed. Should be overriden to 
        add interactivity to the automaton.

        Parameters:
        event : pygame.event
            The event to process
        camera : Camera
            The camera object. Need for the call to self.get_mouse_state.
        """
        pass

    @property
    def worldmap(self):
        """
        Converts self._worldmap to a numpy array, and returns it in a pygame-plottable format (W,H,3).

        Should be overriden only if you use another format for self._worldmap, instead of a torch (3,H,W) tensor.
        """
        return (255 * self._worldmap.permute(2, 1, 0)).detach().cpu().numpy().astype(dtype=np.uint8)

    def get_mouse_state(self, camera):
        """
        Helper function that returns the current mouse state. 
        NOTE : All three mouse button will be considered not pressed if CTRL is pressed.
        This is to prevent interactivity to mix with the camera movement, which uses CTRL+mouse buttons.

        Args:
        camera : Camera
            The camera object. Needed to convert mouse positions to world coordinates.
        
        Returns: mouse_state, an EasyDict with keys :
            x : x position in the CA world (access also as mouse_state.x)
            y : y position in the CA world (access also as mouse_state.y)
            left : True if left mouse button is pressed (access also as mouse_state.left)
            right : True if right mouse button is pressed (access also as mouse_state.right)
            middle : True if middle mouse button is pressed (access also as mouse_state.middle)
        """
        left, middle, right = pygame.mouse.get_pressed()
        mouse_x, mouse_y = camera.convert_mouse_pos(pygame.mouse.get_pos())
        mods = pygame.key.get_mods()
        ctrl_pressed = mods & (pygame.KMOD_LCTRL | pygame.KMOD_RCTRL)
        
        # If CTRL is pressed, force all mouse buttons to be considered not pressed
        if ctrl_pressed:
            left = middle = right = 0
        return EasyDict({"x":mouse_x, "y":mouse_y, 'left': left==1, 'right': right==1, 'middle': middle==1})


    def get_help(self):
        doc = self.__doc__
        process = self.process_event.__doc__
        if(doc is None):
            doc = "No description available"
        if(process is None):
            process = "No interactivity help available"
        return dedent(doc), dedent(process)
