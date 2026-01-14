import pygame_gui
from .BaseComponent import BaseComponent
from .BoxHolder import BoxHolder

class HorizContainer(BaseComponent):
    """
        Horizontal container for automatically stacking BaseComponents horizontally.
    """

    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), max_size=None, rel_padding=0.01):
        """
        Initializes the HorizContainer component.

        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the container.
            max_size (tuple, optional): Maximum size for the component (height, width).
            rel_padding (float): Relative padding between stacked components.
            visible (bool): Whether the container is visible.
        """
        super().__init__(manager, parent, rel_pos, rel_size, max_size=max_size)

        self.h_holder = BoxHolder(
            manager=self.manager,
            parent=self.parent,
            rel_pos=rel_pos,
            rel_size=rel_size,
            max_size=max_size)

        self.padding = rel_padding
        self.components = []

        self.register_main_component(self.h_holder)


    def add_component(self, component: BaseComponent):
        """
        Adds a BaseComponent to the horizontal container.
        Note that the component's parent should be set to this HorizContainer.

        Args:
            component (BaseComponent): The component to add.
        """
        padding = self.padding*self.w
        if len(self.components) > 0:
            last_el = self.components[-1].main_element
            component.set_anchors({
                'left': 'left',
                'left_target': last_el,
                'centery': 'centery'
            })
            component.rel_pos = (padding, 0) # Relative to last component
            component.main_element.set_relative_position((padding, 0))
        else:
            component.set_anchors({
                'left': 'left',
                'centery': 'centery'
            })
            component.rel_pos = (0, 0) # Relative to container left
            component.main_element.set_relative_position((0, 0))

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
        Renders the HorizContainer component with correct positioning and size.
        """
        super().render()
        componentos = self.components
        self.components = []
        for comp in componentos:
            self.add_component(comp)
