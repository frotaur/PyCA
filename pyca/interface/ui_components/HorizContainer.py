from .UIComponent import UIComponent
from .BoxHolder import BoxHolder

class HorizContainer(BoxHolder):
    """
        Horizontal container for automatically stacking BaseComponents horizontally.
    """

    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), z_pos=1, max_size=None, rel_padding=0.01, resize=True):
        """
        Initializes the HorizContainer component.

        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the container.
            z_pos (int): Z-position for rendering order. Higher values are rendered on top.
            max_size (tuple, optional): Maximum size for the component (height, width).
            rel_padding (float): Relative padding between stacked components.
            resize (bool): Whether to allow resizing of the container. If True, the container can be resized horizontally.

        """
        if resize:
            resize_dirs = ['right']
        else:
            resize_dirs = []
        super().__init__(manager, parent, rel_pos, rel_size, z_pos=z_pos, max_size=max_size, resize_dirs=resize_dirs)

        self.padding = rel_padding
        self.components = []


    def add_component(self, component: UIComponent):
        """
        Adds a BaseComponent to the horizontal container.
        Note that the component's parent should be set to this HorizContainer.

        Args:
            component (BaseComponent): The component to add.
        """
        if len(self.components) > 0:
            last_el = self.components[-1].main_element
            component.set_anchors({
                'left': 'left',
                'left_target': last_el,
                'centery': 'centery'
            })
            component.rel_pos = (self.padding, 0) # Relative to last component
            # component.main_element.set_relative_position((padding, 0))
        else:
            component.set_anchors({
                'left': 'left',
                'centery': 'centery'
            })
            component.rel_pos = (0, 0) # Relative to container left
            # component.main_element.set_relative_position((0, 0))

        self.components.append(component)
        component.rebuild()

    def remove_component(self, component: UIComponent, kill=True):
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
            raise ValueError("Tried to remove a component that is not in this HorizContainer.")
        
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
