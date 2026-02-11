from .TextBox import TextBox
from .Slider import Slider
from .TextLabel import TextLabel
from .BoxHolder import BoxHolder
from .UIComponent import UIComponent


class LabeledSlider(BoxHolder):
    """
        A (optionally labeled) slider component with a text label showing the current value.
    """
    def __init__(self, min_value, max_value, manager, parent=None, rel_pos=(0,0),
                 rel_size=(0.05,0.3), title=None,tick_size=1., initial_value=None):
        """
        Initializes the LabeledSlider component.
        
        Args:
            min_value (float): Minimum value of the slider.
            max_value (float): Maximum value of the slider.
            manager (pygame_gui.UIManager): The UI manager.
            parent (BaseComponent, optional): Parent component. Defaults to None.
            rel_pos (tuple, optional): Relative position (y, x) in [0,1]. Defaults to (0,0).
            rel_size (tuple, optional): Relative size (height, width) in [0,1]. Defaults to (0.05,0.3).
            title (str, optional): Title label for the slider. Defaults to None.
            tick_size (float, optional): Increment size for the slider. Defaults to 1.
            initial_value (float, optional): Initial value of the slider. Defaults to None.
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        # Compute the configuration inside the BoxHolder
        title_height_fraction = 0.1 if title else 0.0
        label_width_fraction = 0.15
        slider_pos = (0, title_height_fraction)
        slider_size = (1 - title_height_fraction, 1 - label_width_fraction)
        label_pos = (1 - label_width_fraction, title_height_fraction)
        label_size = (1 - title_height_fraction, label_width_fraction)

        self.title_label = None
        if(title is not None):
            # Create the title label
            self.title_label = TextLabel(
                text=title,
                manager=manager,
                parent=self,
                rel_pos=(0, 0),
                rel_size=(title_height_fraction, 1.0)
            )
        
        self.slider = Slider(
            min_value=min_value,
            max_value=max_value,
            manager=manager,
            parent=self,
            rel_pos=slider_pos,
            rel_size=slider_size,
            tick_size=tick_size,
            initial_value=initial_value)
        
        self.value_label = TextLabel(
            text=str(self.slider.value),
            manager=manager,
            parent=self,
            rel_pos=label_pos,
            rel_size=label_size,
            font_size=8)


    @property
    def value(self):
        """
        Returns the current value of the slider.
        """
        return self.slider.value
    
    @value.setter
    def value(self, new_value):
        """
        Sets the current value of the slider and updates the label.
        
        Args:
            new_value (float): The new value to set.
        """
        self.slider.value = new_value
        self.value_label.text = f'{new_value:.1f}'
    
    def handle_event(self, event):
        """
        Handles events for the LabeledSlider component.
        
        Args:
            event (pygame.event.Event): The event to handle.
        """
        if self.slider.handle_event(event):
            # Update the value label if the slider value changed
            self.value = self.slider.value
            return True
        return False