import numpy as np
import torch
from textwrap import dedent
import pygame
from easydict import EasyDict
from ..interface.ui_components.BaseComponent import BaseComponent

AUTOMATAS = {}

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
    
        self._components = []  # List of GUI components associated to this automaton
        self._custom_size_flags =[]  # List of flags indicating if the component has a custom size
        self._components_fract_pos = (0.7, 0.15)

        self._changed_components = []

        # Obtain (H,W) tensors, containing the x (i.e. width, or column index) and y (i.e. height, or row index) coordinates of each pixel
        self.X, self.Y = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing='xy') 
    
    def __init_subclass__(cls, **kwargs):
            """
                Automagically registers subclasses of Automaton in the
                automaton registry.
            """
            super().__init_subclass__(**kwargs)

            AUTOMATAS[cls.__name__] = cls

    def register_component(self, component:BaseComponent, custom_size=False):
        """
        Registers a GUI component to this automaton. The componenents
        will be rendered on the right side of the screen, and the events
        will be handled automatically.

        Parameters:
        component : BaseComponent
            The component to register.
        custom_size : bool
            If True, preserves the component's set fractional size,
            otherwise, will be resized.
        """
        assert isinstance(component, BaseComponent), "component must be an instance of BaseComponent"
        self._components.append(component)
        self._custom_size_flags.append(custom_size)
        self._place_components(None)  # Place components correctly

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        """
        This method should be overriden. It should update the self._worldmap tensor,
        drawing the current state of the CA world. self._worldmap is a torch tensor of shape (3,H,W).
        If you choose to use another format, you should override the worldmap property as well.
        """
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')
    
    def draw_components(self, screen):
        """
        Draws all registered components to the screen.
        """
        for component in self._components:
            screen = component._draw(screen)
        
        return screen

    def _place_components(self, screen):
        """
            Called only once at the beginning. Puts the correct fractional positions
            so that components are placed on a vertical column, starting at self._components_fract_pos.
        """
        # Constants for component placement
        VERTICAL_SPACING = 0.01  # Fractional spacing between components
        HORIZONTAL_MARGIN = 0.01  # Fractional margin from the right edge
        current_y = self._components_fract_pos[1]  # Start at the specified y position
        x_position = self._components_fract_pos[0]  # Use the specified x position
        
        for i,component in enumerate(self._components):
            # Set the component's fractional position
            component.f_pos = (x_position, current_y)
            if(not self._custom_size_flags[i]):
                component.f_size = (component.f_size[0], 1-x_position-HORIZONTAL_MARGIN)  # Set a fixed width for all components
            # Move to the next position (current y + component height + spacing)
            current_y += component.f_size[0] + VERTICAL_SPACING

    def _process_event_focus_check(self, event, camera=None):
        """
        Like process event, but checks first if we can handle the event, or
        if it is being captured by (focused) component.
        """
        self._process_gui_event(event) # Always try to process GUI events
        if not BaseComponent.get_focus_manager().should_process_event(event):
            #Finish execution here !! 
            return

        self.process_event(event, camera)

        self._changed_components = [] # Reset changed components after processing
    
    def set_components_fract_pos(self, pos):
        """
        Sets the fractional position of the start of the GUI components
        """
        self._components_fract_pos = pos
        self._place_components(None)  # Re-place components correctly

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
    
    def _process_gui_event(self, event):
        """
        Internal. When implemented, will simply pass the event to all defined GUI components,
        so they can handle it if needed. No need to override it if I do things correctly
        """
        
        for component in self._components:
            if(component._handle_event(event)):
                self._changed_components.append(component)

    @property
    def worldmap(self):
        """
        Converts self._worldmap to a numpy array, and returns it in a pygame-plottable format (W,H,3).

        Should be overriden only if you use another format for self._worldmap, instead of a torch (3,H,W) tensor.
        """
        return (255 * self._worldmap.permute(2, 1, 0)).detach().cpu().numpy().astype(dtype=np.uint8)

    @property
    def worldsurface(self):
        """
            Converts self.worldmap to a pygame surface.

            Can be overriden for more complex drawing operations, 
            such as blitting sprites.
        """
        return pygame.surfarray.make_surface(self.worldmap)
    
    @property
    def changed_components(self):
        """
        Returns a list of components that have changed their state at the current event loop iteration.
        It's likely you will want to do something with the value/state of the components in this list!
        """
        return self._changed_components

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
            inside : True if mouse is inside the CA world, False otherwise (access also as mouse_state.inside)
        """
        left, middle, right = pygame.mouse.get_pressed()
        mouse_x, mouse_y = camera.convert_mouse_pos(pygame.mouse.get_pos())
        mods = pygame.key.get_mods()
        ctrl_pressed = mods & (pygame.KMOD_LCTRL | pygame.KMOD_RCTRL)
        
        # If CTRL is pressed, force all mouse buttons to be considered not pressed
        if ctrl_pressed:
            left = middle = right = 0
        return EasyDict({"x":mouse_x, "y":mouse_y, 'left': left==1, 'right': right==1, 'middle': middle==1, 'inside':camera.mouse_in_border(pygame.mouse.get_pos())})


    def get_help(self):
        doc = self.__doc__
        process = self.process_event.__doc__
        if(doc is None):
            doc = "No description available"
        if(process is None):
            process = "No interactivity help available"
        return dedent(doc), dedent(process)

    def name(self):
        """
        Returns the name of the automaton. By default it is the class name,
        but can be overridden in subclasses to provide a more descriptive name.
        """
        return self.__class__.__name__
    
    def get_string_state(self):
        """
        Can be overriden to return a string that gives some live information
        about the model. It is queried each draw call, and displayed on the screen.
        """
        return "Live stuff!"

    def toggle_components(self):
        """
        Toggles the visibility of all registered components.
        """
        for component in self._components:
            component.toggle_visibility()

    def set_components_visibility(self, visible:bool):
        """
        Sets the visibility of all registered components.

        Parameters:
        visible : bool
            If True, shows all components. If False, hides all components.
        """
        for component in self._components:
            component.visible = visible 