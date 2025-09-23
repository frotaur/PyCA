import pygame
from .GlobalFocusManager import GlobalFocusManager

class BaseComponent:
    """
    Base class for UI components in the PyCA interface.
    """
    _global_focus_manager = None

    @classmethod
    def get_focus_manager(cls):
        if cls._global_focus_manager is None:
            cls._global_focus_manager = GlobalFocusManager()

        return cls._global_focus_manager

    def __init__(self,fract_position=(0,0), fract_size=(0.1,0.1), max_size=None):
        """
        Initializes the base component with screen size, fractional position, and size.
        
        Args:
            fract_position (tuple): Fractional position in [0,1] of the component (x (widthloc), y (heighloc)).
            fract_size (tuple): Fractional size in [0,1] of the component (height, width).
            max_size (tuple, optional): Maximum size for the component (height, width).
        """
        self.sH, self.sW = None, None # Is set on draw
        self.f_pos = fract_position
        self.f_size = fract_size

        self.max_size = max_size if max_size else (float('inf'), float('inf'))

        self.visible = True

    @property
    def size(self):
        """
        Returns the size of the component based on the screen size and fractional size.
        """
        if(self.sH is None or self.sW is None):
            print(f"Warning: screen size not set. Returning 0,0. in {self.__class__.__name__}")
            return (0,0)
    
        return (min(int(self.sH * self.f_size[0]), self.max_size[0]), min(int(self.sW * self.f_size[1]), self.max_size[1]))

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
        Returns the position (width_loc, height_loc) of the component based on the screen size and fractional position.

        Note the convention different from sizes, which are (height, width). This is to make it more seamless to use with pygame,
        and also more natural (x, then y), but still matching pytorch conventions which are tensors in (H,W). NOTE : need to decide
        in the future, feels a bit too mixed.
        """
        if(self.sH is None or self.sW is None):
            print(f"Warning: screen size not set. Returning 0,0. in {self.__class__.__name__}")
            return (0,0)
        
        return (int(self.sW * self.f_pos[0]), int(self.sH * self.f_pos[1]))

    @property
    def x(self):
        """
        Returns the x-coordinate (location along width) of the component's position.
        Starts on the left.
        """
        return self.position[0]
    
    @property
    def y(self):
        """
        Returns the y-coordinate (location along height) of the component's position.
        Starts on the top.
        """
        return self.position[1]

    def set_screen_size(self, h,w):
        """
        Sets the screen size for the component.
        
        Args:
            h (int): Height of the screen.
            w (int): Width of the screen.
        """
        self.sH, self.sW = h, w

    def _draw(self, screen):
        """
        Method to actually call to draw the component, setting screen size and checking visibility.


        TODO: In the future, this code should automatically be pre-pended to draw. 
        This can be done with __init_subclass__, but for now, I leave it like this.
        """
        if not self.visible:
            return screen
        
        newW, newH = screen.get_size()
        if self.sW != newW or self.sH != newH:
            self.set_screen_size(newH, newW)
            self.render()
        
        return self.draw(screen)
    
    def draw(self, screen : pygame.Surface) -> pygame.Surface:
        """
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
    
    def render(self):
        """
        Renders the component. Inside render, you can assume that *self.sH* and *self.sW*
        are set, and so *self.position* and *self.size* can be accessed (equivalently,
        *self.h*, *self.w*, *self.x*, *self.y*). You should use those
        to render the full component on a surface (or many surfaces), that will then be
        drawn (blitted) on the screen in the draw method. This allows rendering to be called
        only when the screen size changes, and we need to update the component's position or size.

        This function should be subclassed, even if no special rendering is needed, in this case, just pass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _handle_event(self, event, parent_mouse_pos=None):
        """
        This function should be called on to handle an event. It should
        NOT be overridden, instead, override handle_event to add custom event handling.

        TODO: In the future, this code should automatically be pre-pended to handle_event.
        This can be done with __init_subclass__, but for now, I leave it like this.
        """
        if not self.visible:
            return False
        
        # Ideally, I could check focus here and return if not focused, but in practice,
        # I cannot do it... Indeed, it would break nested components, as the parent
        # may not have focus, but the child may. In this case, we would return early,
        # and the child would never get the event. So, I leave it to the component itself
        # to check if it is focused or not, if needed. Maybe I will figure out a better way in the future.

        if(parent_mouse_pos is None):
            return self.handle_event(event) # Retrocompatible with handle_event without parent pos
        else:
            return self.handle_event(event, parent_mouse_pos)

    def handle_event(self, event: pygame.event.Event, parent_mouse_pos=None) -> bool:
        """
        Handles an event for the component. Should be implement in subclasses.
        It should return True if some important value of the component has changed. 
        This is so the user can, in this case, update its state accordingly, 
        and avoid unnecessary updates.
        
        Args:
            event (pygame.event.Event): The event to handle.
            parent_mouse_pos (tuple, optional): Position of mouse in the parent's coordinates.
            Use with get_mouse_pos for collisions.
        Returns:
            bool: True if some value of the component changed, False otherwise.
        """
        return False
    

        
    def get_mouse_pos(self, event, parent_mouse_pos):
        """
        Converts a mouse position in 'parent' coordinates to local component coordinates.
        
        Args:
            mouse_pos (tuple): Mouse position in 'parent' coordinates.
        """
        if(parent_mouse_pos is None):
            parent_mouse_pos = event.pos if hasattr(event, 'pos') else None
        
        return (parent_mouse_pos[0]-self.x, parent_mouse_pos[1]-self.y) if parent_mouse_pos is not None else None

    def request_keyboard_focus(self):
        """
        Requests keyboard focus for this component.
        """
        self.get_focus_manager().request_keyboard_focus(self)
    
    def release_keyboard_focus(self):
        """
        Releases keyboard focus for this component.
        """
        self.get_focus_manager().release_keyboard_focus(self)
    
    def am_focused(self):
        """
        Returns True if this component is currently focused, False otherwise.
        """
        return self.get_focus_manager().am_focused(self)

    def on_focus_lost(self):
        """
        Called when the component loses focus. Can be overridden in subclasses.
        """
        pass

    def on_focus_gained(self):
        """
        Called when the component gains focus. Can be overridden in subclasses.
        """
        pass

    def toggle_visibility(self):
        """
        Toggles the visibility state of the component.
        """
        self.visible = not self.visible