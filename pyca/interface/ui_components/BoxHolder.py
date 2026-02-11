import pygame_gui
from .UIComponent import UIComponent
import pygame
from pygame_gui.core.ui_container import UIContainer
from pygame_gui.elements.ui_panel import UIPanel
from pygame_gui.elements.ui_auto_resizing_container import UIAutoResizingContainer

class BoxHolder(UIComponent):
    """
        Transparent container, used only for relative positioning of other components.
    """
    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), z_pos=1, max_size=None, resize_dirs=[]):
        """
        Initializes the BoxHolder component.

        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the box holder.
            z_pos (int): Z-position for rendering order. Higher values are rendered on top.
            max_size (tuple, optional): Maximum size for the component (height, width).
            resize_dirs (list, optional): List of directions to allow resizing. Options are 'left', 'right', 'top', 'bottom'.
        """
        super().__init__(manager, parent, rel_pos, rel_size, max_size=max_size)
    
        # if(not visible):
        #     self.box = UIContainer(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
        #                              manager=self.manager)
        # else:
        #     self.box = UIPanel(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
        #                              manager=self.manager)
        
        self.box = UIAutoResizingContainer(
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            resize_left='left' in resize_dirs,
            resize_right='right' in resize_dirs,
            resize_top='top' in resize_dirs,
            resize_bottom='bottom' in resize_dirs,
            manager=self.manager,
            container=parent.container if parent is not None else None,
            starting_height=z_pos
        )

        self.register_main_component(self.box)

    def render(self):
        """
        Renders the BoxHolder component with correct positioning and size.
        """
        # Update window position and size
        self.box.set_relative_position((self.x, self.y))
        self.box.set_dimensions((self.w, self.h))
        
        self.box.rebuild()
