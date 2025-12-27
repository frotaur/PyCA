import pygame_gui
from .BaseComponent import BaseComponent
import pygame
from pygame_gui.core.ui_container import UIContainer
from pygame_gui.elements.ui_panel import UIPanel

class BoxHolder(BaseComponent):
    """
        Transparent container, used only for relative positioning of other components.
    """
    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), max_size=None, visible=False):
        """
        Initializes the BoxHolder component.

        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the box holder.
            max_size (tuple, optional): Maximum size for the component (height, width).
        """
        super().__init__(manager, parent, rel_pos, rel_size, max_size)
    
        if(not visible):
            self.box = UIContainer(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
                                     manager=self.manager)
        else:
            self.box = UIPanel(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
                                     manager=self.manager)
    
    @property
    def container(self):
        """
        Returns the underlying UIContainer.
        """
        return self.box.get_container()

    def render(self):
        """
        Renders the BoxHolder component with correct positioning and size.
        """
        super().render() 

        # Update window position and size
        self.box.set_relative_position((self.x, self.y))
        self.box.set_dimensions((self.w, self.h))
    