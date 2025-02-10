import numpy as np
import torch
from textwrap import dedent
import pygame
from easydict import EasyDict


class Automaton:
    """
    Class that internalizes the rules and evolution of
    a cellular automaton. It has a step function
    that makes one timestep of evolution.
    By convention, the world tensor has shape
    (3,H,W). It contains float values between 0 and 1, which
    are mapped to [0, 255] and casted to a uint8 numpy array,
    which is compatible with pygame.
    """

    def __init__(self, size):
        """
        Parameters :
        size : 2-uple (H,W)
            Shape of the CA world
        """
        self.h, self.w = size
        self.size = size

        self._worldmap = torch.zeros((3, self.h, self.w))  # (3,H,W), contains a 2D 'view' of the CA world

        self.right_pressed = False
        self.left_pressed = False

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        """
        This function should update the self._worldmap tensor
        """
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')

    @property
    def worldmap(self):
        """Converts _worldmap to a numpy array, and returns it in a pygame-plottable format (W,H,3).

        Can be overriden if you use another format for self._worldmap, instead of a torch (3,H,W) tensor.
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
        
        Returns: mouse_state an EasyDict with keys :
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

    def process_event(self, event, camera=None):
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

    def get_help(self):
        return dedent(self.__doc__), dedent(self.process_event.__doc__)
