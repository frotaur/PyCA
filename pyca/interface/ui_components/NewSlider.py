from .BaseComponent import BaseComponent
import pygame, pygame_gui
from pygame_gui.elements.ui_horizontal_slider import UIHorizontalSlider

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

        self.slider = UIHorizontalSlider(
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            start_value=self._value,
            value_range=(min_value, max_value),
            manager=self.manager,
            container=self.parent.container if self.parent is not None else None,
            click_increment=tick_size
        )
    
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
        self.slider.set_relative_position((self.x, self.y))
        self.slider.set_dimensions((self.w, self.h))

    def handle_event(self, event: pygame.event.Event, parent_mouse_pos=None) -> bool:
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

class LabeledSlider(BaseComponent):
    """
    A slider component with a text label showing the current value.
    """
    
    def __init__(self, min_value=0, max_value=100, title=None,precision=1,
                  num_steps=100, initial_value=None, label_width_fraction=0.15,
                  fract_position=(0, 0), fract_size=(0.05, 0.3), 
                 bg_color=(50, 50, 50), slider_color=(100, 100, 100), 
                 handle_color=(200, 200, 200),
                 font_path: str = BASE_FONT_PATH):
        """
        Initializes the labeled slider component.
        
        Args:
            min_value (float): Minimum value of the slider.
            max_value (float): Maximum value of the slider.
            num_steps (int): Number of discrete steps between min and max values.
            initial_value (float, optional): Initial value. If None, defaults to min_value.
            precision (int): Number of decimal places to show in the label. Default is 1.
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the labeled slider (height, width).
            bg_color (tuple): Background color of the slider track in RGB format.
            slider_color (tuple): Color of the slider track in RGB format.
            handle_color (tuple): Color of the slider handle in RGB format.
            font_path (str): File path to the font to be used.
        """
        super().__init__(fract_position, fract_size)
        
        self.precision = precision

        # Calculate proportions - label takes about 10% of the width
        # Title takes takes 50% of the height
        title_height_fraction = 0.4 if title else 0.0
        
        slider_width_fraction = 1 - label_width_fraction
        slider_height_fraction = 1 - title_height_fraction
        
        # Create the slider component (takes most of the width)
        slider_f_size = (slider_height_fraction, slider_width_fraction)

        slider_f_pos = (0, title_height_fraction)
        
        self.slider = Slider(
            min_value=min_value, max_value=max_value, num_steps=num_steps,
            initial_value=initial_value, fract_position=slider_f_pos,
            fract_size=slider_f_size, bg_color=bg_color,
            slider_color=slider_color, handle_color=handle_color
        )
        
        # Create font for the label
        self.font = SmartFont(font_path=font_path, fract_font_size=slider_height_fraction*0.8, 
                             min_font_size=8, max_font_size=16)
        
        # Create the label component (takes remaining width)
        label_spacing = 0.02
        label_f_pos = (slider_width_fraction+label_spacing, title_height_fraction)
        label_f_width = label_width_fraction-label_spacing
    
        
        # Initialize with current value
        initial_text = self._format_value(self.slider.value)
        self.label = TextLabel(
            text=initial_text,
            font=self.font,
            fract_position=label_f_pos,
            fract_width=label_f_width,
            color=(230, 230, 230),
            bg_color=(0, 0, 0, 0),  # Transparent background
            h_margin=0.4
        )
        
        if(title is not None):
            title_f_pos = (0, 0)
            self.title_font = SmartFont(font_path=font_path, fract_font_size=title_height_fraction*0.8, 
                                       min_font_size=10, max_font_size=18)
            self.title = TextLabel(
                text=title,
                font=self.title_font,
                fract_position=title_f_pos,
                fract_width=1.0,
                color=(255,89,89),
                bg_color=(0, 0, 0, 0),  # Transparent background
                h_margin=0.
            )
        self.previous_value = self.slider.value

        # Rendered surfaces
        self.full_surface = None

    def _format_value(self, value):
        """Format the value according to the specified precision."""
        return f"{value:.{self.precision}f}"
    
    @property
    def value(self):
        """Get the current value of the slider."""
        return self.slider.value
    
    @value.setter
    def value(self, new_value):
        """Set the value of the slider and update the label."""
        self.slider.value = new_value
        self.label.text = self._format_value(self.slider.value)
        self.previous_value = self.slider.value

    def handle_event(self, event: pygame.event.Event, local_mouse_pos=None) -> bool:
        """
        Handle events for the labeled slider component.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if the slider value changed (on mouse release), False otherwise.
        """
        super().handle_event(event, local_mouse_pos)
        # Handle slider events
        value_changed = self.slider.handle_event(event, self.get_mouse_pos(event, local_mouse_pos))
        
        # Update label if value changed during dragging or on final change
        if abs(self.slider.value - self.previous_value) > 1e-10:
            self.label.text = self._format_value(self.slider.value)
            self.previous_value = self.slider.value
        
        return value_changed
    
    def render(self):
        """
        Render method - no need to render anything as subcomponents handle their own rendering.
        """
        self.full_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

    
    def draw(self, screen: pygame.Surface) -> pygame.Surface:
        """
        Draws the labeled slider component to the screen.
        
        Args:
            screen (pygame.Surface): The surface to draw on.
        
        Returns:
            pygame.Surface: The screen with the component drawn on it.
        """
        # Draw slider and label
        self.full_surface.fill((0, 0, 0, 0))  # Clear with transparent
        if(self.title):
            surface = self.title._draw(self.full_surface)
        surface = self.slider._draw(self.full_surface)
        surface = self.label._draw(surface)
        
        screen.blit(surface, (self.x, self.y))
        
        return screen