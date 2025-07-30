import pygame


class BaseComponent:
    """
    Base class for UI components in the PyCA interface.
    """

    def __init__(self,fract_position, fract_size, max_size=None):
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

    @property
    def size(self):
        """
        Returns the size of the component based on the screen size and fractional size.
        """
        if(self.sH is None or self.sW is None):
            print("Warning: screen size not set. Returning 0,0.")
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
            print("Warning: screen size not set. Returning 0,0.")
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

    def draw(self, screen : pygame.Surface) -> pygame.Surface:
        """
        Draws the component on the given screen. Must be subclassed.
        This basic method only updates screen size, and calls the 'render'
        method if the screen size has changed. In the subclass, simply call
        super().draw(screen), and then use whatever is prepared in the render method
        to draw the component.
        
        Args:
            screen (pygame.Surface): Screen surface.
        
        Returns:
            pygame.Surface: The screen with the component drawn on it.
        """
        newW, newH = screen.get_size()
        if self.sW != newW or self.sH != newH:
            self.set_screen_size(newH, newW)
            self.render()

        return screen
    
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
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handles an event for the component. Must be subclassed. It should return True
        if some important value of the component has changed. This is so the user can,
        in this case, update its state accordingly, and avoid unnecessary updates.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if some value of the component changed, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")