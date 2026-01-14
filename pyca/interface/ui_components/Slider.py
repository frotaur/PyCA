from .BaseComponent import BaseComponent
import pygame, pygame_gui
from pygame_gui.elements.ui_horizontal_slider import UIHorizontalSlider
from pygame_gui.core.ui_container import UIContainer

class Slider(BaseComponent):
    """
    A horizontal slider component with configurable min/max values and steps.
    """
    
    def __init__(self, min_value, max_value, manager, parent=None,rel_pos=(0, 0), rel_size=(0.05, 0.2), tick_size=1, initial_value=None
                 ):
        """
        Initializes the slider component.
        
        Args:
            min_value (float or int): Minimum value of the slider.
            max_value (float or int): Maximum value of the slider.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            tick_size (float or int): Step size between min and max values.
            initial_value (float or int, optional): Initial value. If None, defaults to min_value.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width).
        """
        super().__init__(manager, parent, rel_pos, rel_size)
        self.tick_size = tick_size
        
        # Initialize value
        if initial_value is None:
            self._value = min_value
        else:
            self._value = max(min_value, min(max_value, initial_value))
        
        # Use the UIContainer as a box because the slider is bugged the fuck out
        self.boxer = UIContainer(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
                                      manager=self.manager,
                                      container=self.parent.container if self.parent is not None else None)
        
        self.slider = UIHorizontalSlider(
            relative_rect=pygame.Rect(0, 0, self.w, self.h),
            start_value=self._value,
            value_range=(min_value, max_value),
            manager=self.manager,
            container=self.boxer.get_container(),
            click_increment=tick_size
        )

        self.register_main_component(self.boxer)

    @property
    def value(self):
        """Get the current value of the slider."""
        return self.slider.get_current_value()
    
    @value.setter
    def value(self, new_value):
        """Set the value of the slider."""
        self.slider.set_current_value(new_value)
    
    def render(self):
        """
        Renders the slider component.
        """
        # self.slider.set_relative_position((self.x, self.y))
        self.boxer.set_relative_position((self.x, self.y))
        self.boxer.set_dimensions((self.w, self.h))
        self.slider.set_dimensions((self.w, self.h))

        self.slider.rebuild()
        self.boxer.rebuild()
    

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handles events for the slider component.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if the slider value changed
        """
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.slider:
                return True
        return False
