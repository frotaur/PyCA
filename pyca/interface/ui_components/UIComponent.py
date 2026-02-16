from __future__ import annotations

import pygame
from pygame_gui.core.ui_element import UIElement
from pygame_gui.core.utility import get_default_manager


class UIComponent:
    """
    Base class for UI components in the PyCA interface.
    """
    BASE_FONT_REL_SIZE = 0.015  # Relative font size with respect to screen height
    
    def __init__(self, manager=None, parent = None, rel_pos=(0,0), rel_size=(0.1,0.1), max_size=None):
        """
        Initializes the base component with screen size, fractional position, and size.
        
        Args:
            manager: pygame-gui UIManager instance.
            container: parent UI window, if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x (widthloc), y (heighloc)).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width).
            max_size (tuple, optional): Maximum size for the component (height, width).
            z_pos (int, optional): Z-position for rendering order. Higher values are rendered on top.
            theme_class (str, optional): Theme class for styling
            theme_id (str, optional): Theme ID for styling
        """
        self.manager = manager if manager is not None else get_default_manager()
        self.sW, self.sH = self.manager.window_resolution

        self.parent = parent

        if(self.parent is not None):
            self.parent._register_child_component(self)

        self._rel_pos = rel_pos
        self._rel_size = rel_size
        self.max_size = max_size if max_size else (float('inf'), float('inf'))

        self.main_element = None

        self.base_font_size = 12
        self.child_components = []

        

    @property
    def rel_pos(self):
        """
        Returns the fractional position of the component.
        """
        return self._rel_pos
    
    @rel_pos.setter
    def rel_pos(self, new_rel_pos):
        """
            Sets rel_pos
        """
        self._rel_pos = new_rel_pos
        self._render(force=True) # Force re-render to update position immediately when rel_pos is changed

    @property
    def rel_size(self):
        """
        Returns the fractional size of the component.
        """
        return self._rel_size

    @rel_size.setter
    def rel_size(self, new_rel_size):
        """
            Sets rel_size
        """
        self._rel_size = new_rel_size
        self._render(force=True) # Force re-render to update size immediately when rel_size is changed

    @property
    def font_abs_size(self):
        """
        Returns the absolute font size of the component.
        """
        font_dict = self.get_font()
        if(font_dict is None):
            return None
        
        # base_size = font_dict['font'].point_size # NOTE : FOR NOW, BUGGED IN PYGAME-GUI so workaround
        base_size = self.base_font_size

        return int(self.BASE_FONT_REL_SIZE*min(self.sH, self.sW)*base_size/12.)

    def register_main_component(self, component: UIElement):
        """
        Registers the main ui element.

        Args:
            component (UIElement): The main element to register.
        """
        self.main_element = component

    def _adjust_font_size(self):
        """
        Sets the absolute font size of the component based on the relative font size and screen height.
        """
        # self.font_abs_size = int(self.font_rel_size * self.sH)

        font_dict = self.get_font()
        if(font_dict is not None):
            font, pygame_font = font_dict['font'], font_dict['pygame_font']
            # font.point_size=self.font_abs_size
            pygame_font.set_point_size(self.font_abs_size)
        
        for child in self.child_components:
            child._adjust_font_size()
        
        self.rebuild()
        
    def _register_child_component(self, component: 'UIComponent'):
        """
        Registers a child component. Used for composite components.
        """
        self.child_components.append(component)

    def get_font(self):
        """
        Returns a dictionary with the pygame-gui font and the underlying pygame font.

        Returns:
            dict: {'font': pygame-gui font, 'pygame_font': pygame font}
        """
        if(not hasattr(self.main_element, 'font')):
            return None
        
        return {'font': self.main_element.font, 'pygame_font': self.main_element.font._GUIFontPygame__internal_font}


    def get_font_size(self) -> int:
        """
        Returns the current font size of the component. If the
        component itself has no font, attempts to get it from children.

        Returns:
            int: The current font size.
        """
        pass # Will probably deprecate this
        font_dict = self.get_font()

        if(font_dict is None):
            if(len(self.child_components)>0):
                for child in self.child_components:
                    f_size = child.get_font_size()
                    if(f_size is not None):
                        return f_size
                return None
            else:
                return None
        
        return font_dict['font'].get_point_size()

    @property
    def container(self):
        """
        Return the container of the component itself (not the parent!).
        """
        return self.main_element.get_container()
    
    
    def _reparent(self, reparent_target, new_parent):
        """
        Reparents a given BaseComponent to a new parent.

        Args:
            reparent_target (BaseComponent): The component to reparent.
            new_parent (BaseComponent | None): The new parent component.
        """
        if(reparent_target.parent is not None):
            reparent_target.parent.child_components.remove(reparent_target)

        reparent_target.parent = new_parent

        if(new_parent is not None):
            new_parent._register_child_component(reparent_target)
        
    def set_parent(self, new_parent: 'UIComponent' | None):
        """
        Sets a new parent for the component.

        Args:
            new_parent (BaseComponent | None): The new parent component.
        """

        self._reparent(reparent_target=self, new_parent=new_parent)

        
        # in any case, update container of main element
        new_container = self.parent.container if self.parent is not None else None
        self.main_element.set_container(new_container)
        self._render(force=True)
        
    
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
        self.sW, self.sH = self.manager.window_resolution
        self._adjust_font_size()
    
    def rebuild(self):
        """
        Rebuilds the component. Should be called after changing any property that affects the size or position of the component.
        """
        self._render(force=True)
        self.main_element.rebuild()

    def _render(self, force=False):
        """
        Internal render method called when screen size changes.
        Calls the user-defined render method.

        Args:
            force (bool): If True, forces re-rendering even if size hasn't changed.
        """
        if((self.sW, self.sH) != self.manager.window_resolution): # Window has been resized
            self.set_screen_size()
        elif(not force):
            return
        
        for child in self.child_components:
            child._render(force=force)
        
        self.render()


    def render(self):
        """
        'Renders' the component, i.e. recomputes sizes if necessary. Only redefine if you are using pygame-gui 
        elements directly, in which case you should use *self.position* and *self.size* 
        (equivalently,*self.h*, *self.w*, *self.x*, *self.y*) to resize/reposition the pygame-gui elements passing absolute sizes. 
        This guarantees that on resize,  the elements will be correctly resized, keeping their relative position and size.

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

    @property
    def visible(self) -> bool:
        return self.main_element.visible
    
    @visible.setter
    def visible(self, value: bool):
        if value:
            self.main_element.show()
        else:
            self.main_element.hide()
    
    def toggle_visibility(self):
        """
        Toggles the visibility state of the component.
        """
        self.visible = not self.visible

    def set_anchors(self, anchors):
        """
        Sets the anchors for the component's main element.
        Can be overridden in subclasses for elements with internal containers.

        Args:
            anchors (dict): A dictionary of anchors defining what the relative rect is relative to.
        """
        self.main_element.set_anchors(anchors)