import pygame_gui
from .BaseComponent import BaseComponent
from .BoxHolder import BoxHolder

class VertContainer(BaseComponent):
    """
        Vertical container for automatically stacking BaseComponents vertically. It 
        auto-resizes downwards to hold exactly all components, while its width is fixed.
    """

    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), max_size=None, rel_padding=0.01):
        """
        Initializes the VertContainer component.

        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the container.
            max_size (tuple, optional): Maximum size for the component (height, width).
            rel_padding (float): Relative padding between stacked components.
        """
        super().__init__(manager, parent, rel_pos, rel_size,max_size= max_size)

        self.v_holder = BoxHolder(
            manager=self.manager,
            parent=self.parent,
            rel_pos=rel_pos,
            rel_size=rel_size,
            max_size=max_size,
            resize_dirs=['bottom'])
        
        self.padding = rel_padding
        self.components = []

        self.register_main_component(self.v_holder)

    
    def add_component(self, component: BaseComponent):
        """
        Adds a BaseComponent to the vertical container.
        Note that the component's parent should be set to this VertContainer.

        Args:
            component (BaseComponent): The component to add.
        """
        padding = self.padding*self.h
        if len(self.components) >0 :
            last_el = self.components[-1].main_element
            component.main_element.set_anchors({
                'top': 'top',
                'top_target': last_el,
                'centerx': 'centerx'
            })
            component.rel_pos = (0, self.padding) # Relative to last component
            # WORKAROUND: set_anchors() doesn't work properly unless we also call set_relative_position()
            # This forces pygame_gui to recalculate the position based on the anchors
            component.main_element.set_relative_position((0, padding))
        else:
            component.main_element.set_anchors({
                'top': 'top',
                'centerx': 'centerx'
            })
            component.rel_pos = (0, 0) # Relative to container top
            component.main_element.set_relative_position((0, 0))

        component.set_parent(self)

        self.components.append(component)
        component.main_element.rebuild()

    def remove_component(self, component: BaseComponent, kill=True):
        """
        Removes a BaseComponent from the vertical container.
        Kills its main element and unsets its parent.

        Args:
            component (BaseComponent): The component to remove.
        """
        if component in self.components:
            self.components.remove(component)
            component.set_parent(None)
            if kill:
                component.main_element.kill()
            else:
                print("Warning: Tried to remove a component that is not in this VertContainer.")
        else:
            raise ValueError("Tried to remove a component that is not in this VertContainer.")
        
        self.render()

    def render(self):
        """
        Renders the VertContainer component with correct positioning and size.
        """
        componentos = self.components
        self.components=[]
        for comp in componentos:
            self.add_component(comp)   