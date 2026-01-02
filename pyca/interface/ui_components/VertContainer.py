import pygame_gui
from .BaseComponent import BaseComponent
from .BoxHolder import BoxHolder

class VertContainer(BaseComponent):
    """
        Vertical container for automatically stacking BaseComponents vertically.
    """

    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), max_size=None, rel_padding=0.01, visible=False):
        """
        Initializes the VertContainer component.

        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the container.
            max_size (tuple, optional): Maximum size for the component (height, width).
            rel_padding (float): Relative padding between stacked components.
            visible (bool): Whether the container is visible.
        """
        super().__init__(manager, parent, rel_pos, rel_size,max_size= max_size)

        self.v_holder = BoxHolder(
            manager=self.manager,
            parent=self.parent,
            rel_pos=rel_pos,
            rel_size=rel_size,
            max_size=max_size,
            visible=visible)
        
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
        if len(self.components) >0 :
            last_el = self.components[-1].main_element
            component.main_element.set_anchors({
                'top': 'top',
                'top_target': last_el,
                'centerx': 'centerx'
            })
            component.rel_pos = (0, self.padding) # Relative to last component
        else:
            component.main_element.set_anchors({
                'top': 'top',
                'centerx': 'centerx'
            })
            component.rel_pos = (0, 0) # Relative to container top
        
        self.components.append(component)
        component.main_element.rebuild()

    def render(self):
        """
        Renders the VertContainer component with correct positioning and size.
        """
        super().render()
        componentos = self.components
        self.components=[]
        for comp in componentos:
            self.add_component(comp)   
        
    def _add_component(self, component: BaseComponent):
        """
        Adds a BaseComponent to the vertical container using computed positions instead of anchors.
        Note that the component's parent should be set to this VertContainer.

        Args:
            component (BaseComponent): The component to add.
        """
        # Get container dimensions
        container_rect = self.v_holder.main_element.get_abs_rect()
        container_width = container_rect.width

        # Calculate padding in pixels
        padding_pixels = self.padding * container_rect.height

        # Calculate vertical position
        if len(self.components) > 0:
            # Position below the last component
            last_component = self.components[-1]
            last_rect = last_component.main_element.get_relative_rect()
            y_pos = last_rect.bottom + padding_pixels - container_rect.y
        else:
            # First component, position at top
            y_pos = 0

        # Get component dimensions
        component_rect = component.main_element.get_relative_rect()
        component_width = component_rect.width

        # Center horizontally
        x_pos = (container_width - component_width) / 2

        # Set the component's position
        component.main_element.set_relative_position((x_pos, y_pos))

        self.components.append(component)
        component.main_element.rebuild()

