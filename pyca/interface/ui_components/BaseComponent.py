from __future__ import annotations

import pygame
from pygame_gui.core.ui_element import UIElement

class BaseComponent:
    """
    Base class for UI components in the PyCA interface.
    """

    def __init__(self, manager, parent = None, rel_pos=(0,0), rel_size=(0.1,0.1), max_size=None):
        """
        Initializes the base component with screen size, fractional position, and size.
        
        Args:
            manager: pygame-gui UIManager instance.
            container: parent UI window, if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x (widthloc), y (heighloc)).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width).
            max_size (tuple, optional): Maximum size for the component (height, width).
        """
        self.manager = manager
        self.sW, self.sH = self.manager.window_resolution

        self.parent = parent

        self.rel_pos = rel_pos
        self.rel_size = rel_size
        self.max_size = max_size if max_size else (float('inf'), float('inf'))

        self.visible = True

        self.main_component = None
    
    def register_main_component(self, component: UIElement | 'BaseComponent'):
        """
        Registers the main component. Can be a pygame-gui UIElement (for core components) 
        or another BaseComponent (for composite components).
        """
        if isinstance(component, BaseComponent):
            self.main_component = component.main_component
        else:
            self.main_component = component

    @property
    def container(self):
        """
        Return the container of the component itself (not the parent!).
        """
        return self.main_component.get_container()
    
    @property
    def parent_size(self):
        """
        Returns the size of the parent container, or the window size if no container.
        """
        if(self.parent is None):
            sW, sH = self.manager.window_resolution
            return (sH, sW)
        else:
            return self.parent.size

    @property
    def parent_position(self):
        """
        Returns the position of the parent container, or (0,0) if no container.
        """
        if(self.parent is None):
            return (0,0)
        else:
            return self.parent.position

    @property
    def size(self):
        """
        Returns the size of the component based on parent size and fractional size.
        """
        pH, pW = self.parent_size
        h = min(int(pH * self.rel_size[0]), self.max_size[0])
        w = min(int(pW * self.rel_size[1]), self.max_size[1])
        # for dynamic sizing in pygame-gui
        h = h if h>0 else -1 
        w = w if w>0 else -1
        
        return (h, w)

    @property
    def h(self):
        """
        Returns the height of the component based on the screen size and fractional size.
        """
        return self.size[0]

    @property
    def w(self):
        """
        Returns the width of the component based on the screen size and fractional size.
        """
        return self.size[1]
    
    @property
    def position(self):
        """
        Returns the relative position of the component (x, y) in pixels, based on the parent size.

        Note the convention different from sizes, which are (height, width). This is to make it more seamless to use with pygame,
        and also more natural (x, then y), but still matching pytorch conventions which are tensors in (H,W). NOTE : need to decide
        in the future, feels a bit too mixed.

        NOTE: REVIEW THE W, H CONVENTIONS
        """
        pH, pW = self.parent_size

        return (int(pW * self.rel_pos[0]), int(pH * self.rel_pos[1]))

    @property
    def x(self):
        """
        Returns the x-coordinate (location along width) of the component's (relative) position.
        Starts on the left.
        """
        return self.position[0]
    
    @property
    def y(self):
        """
        Returns the y-coordinate (location along height) of the component's (relative) position.
        Starts on the top.
        """
        return self.position[1]

    def set_screen_size(self):
        """
        Sets the screen size for the component. Takes it from the manager.
        """
        self.sH, self.sW = self.manager.window_resolution
    
    def draw(self) -> None:
        """
        Draws the compo
        Draws the component on the given screen. Must be subclassed.
        When called, screen size is automatically updated, and the 'render'
        method is called if the screen size changed. In this method, use 
        whatever is prepared in the render method to draw the component.
        
        Args:
            screen (pygame.Surface): Screen surface.
        
        Returns:
            pygame.Surface: The screen with the component drawn on it.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    def _render(self):
        """
        Internal render method called when screen size changes.
        Calls the user-defined render method.
        """
        if((self.sH, self.sW) != self.manager.window_resolution): # Window has been resized
            self.set_screen_size()
        else:
            return
        
        self.render()

    def render(self):
        """
        Renders the component. If you are using base components that are already defined, you need only call
        the .render() method for each of the used components. If you are using pygame-gui elements directly,
        you should use *self.position* and *self.size* (equivalently,*self.h*, *self.w*, *self.x*, *self.y*) 
        to define the pygame-gui elements you need, passing absolute sizes. This guarantees that on resize, 
        the elements will be correctly resized, keeping their relative position and size.

        NOTE: Make _render auto-wrap this method so I don't have to juggle between them.
        """
        pass


    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handles an event for the component. Should be implemented in subclasses.
        See pygame-gui documentation for event types.
        It should return True if some important value of the component has changed. 
        This is so the user can, in this case, update its state accordingly, 
        and avoid unnecessary updates. --> We'll see if we keep this design choice.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if some value of the component changed, False otherwise.
        """
        return False

    def toggle_visibility(self):
        """
        Toggles the visibility state of the component.
        """
        self.visible = not self.visible
        if self.visible:
            self.main_component.show()
        else:
            self.main_component.hide()