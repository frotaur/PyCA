from .BaseComponent import BaseComponent
from .SmartFont import SmartFont
from .TextLabel import TextLabel
from ..files import BASE_FONT_PATH
import pygame


class Slider(BaseComponent):
    """
    A horizontal slider component with configurable min/max values and steps.
    """
    
    def __init__(self, min_value=0, max_value=100, num_steps=100, initial_value=None,
                 fract_position=(0, 0), fract_size=(0.05, 0.2), 
                 bg_color=(50, 50, 50), slider_color=(100, 100, 100), 
                 handle_color=(200, 200, 200)):
        """
        Initializes the slider component.
        
        Args:
            min_value (float): Minimum value of the slider.
            max_value (float): Maximum value of the slider.
            num_steps (int): Number of discrete steps between min and max values.
            initial_value (float, optional): Initial value. If None, defaults to min_value.
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the slider (height, width).
            bg_color (tuple): Background color of the slider track in RGB format.
            slider_color (tuple): Color of the slider track in RGB format.
            handle_color (tuple): Color of the slider handle in RGB format.
        """
        super().__init__(fract_position, fract_size)
        
        self.min_value = min_value
        self.max_value = max_value
        self.num_steps = num_steps
        self.step_size = (max_value - min_value) / num_steps
        
        # Initialize value
        if initial_value is None:
            self._value = min_value
        else:
            self._value = max(min_value, min(max_value, initial_value))
        
        # Colors
        self.bg_color = bg_color
        self.slider_color = slider_color
        self.handle_color = handle_color
        self.border_color = tuple(min(bg + 30, 255) for bg in bg_color)
        
        # State
        self.is_dragging = False
        self.mouse_was_pressed_on_slider = False
        self.drag_offset = 0
        self.value_changed = False
        self.previous_value = self._value
        
        # Rendered surfaces
        self.slider_surface = None
        self.slider_rect = None
        self.handle_rect = None
        
        # Dimensions (as fractions of component size)
        self.track_height_fraction = 0.3  # Height of the track relative to component height
        self.handle_width_fraction = 0.1   # Width of handle relative to component width
        self.handle_height_fraction = 0.8  # Height of handle relative to component height
    
    @property
    def value(self):
        """Get the current value of the slider."""
        return self._value
    
    @value.setter
    def value(self, new_value):
        """Set the value of the slider."""
        self._value = max(self.min_value, min(self.max_value, new_value))
        if hasattr(self, 'slider_surface') and self.slider_surface:
            self.render()
    
    def _value_to_position(self, value):
        """Convert a value to x position on the slider track."""
        if self.max_value == self.min_value:
            return 0
        
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        handle_width = int(self.w * self.handle_width_fraction)
        usable_width = self.w - handle_width
        return int(normalized * usable_width)
    
    def _position_to_value(self, x_pos):
        """Convert x position on the slider track to a value."""
        handle_width = int(self.w * self.handle_width_fraction)
        usable_width = self.w - handle_width
        
        if usable_width <= 0:
            return self.min_value
        
        # Clamp position to valid range
        x_pos = max(0, min(usable_width, x_pos))
        
        normalized = x_pos / usable_width
        raw_value = self.min_value + normalized * (self.max_value - self.min_value)
        
        # Snap to nearest step
        step_index = round((raw_value - self.min_value) / self.step_size)
        step_index = max(0, min(self.num_steps, step_index))
        
        return self.min_value + step_index * self.step_size
    
    def render(self):
        """
        Renders the slider component.
        """
        # Create slider surface
        self.slider_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.slider_rect = pygame.Rect(self.x, self.y, self.w, self.h)
        
        # Draw background
        self.slider_surface.fill(self.bg_color)
        
        # Draw slider track
        track_height = max(2, int(self.h * self.track_height_fraction))
        track_y = (self.h - track_height) // 2
        track_rect = pygame.Rect(0, track_y, self.w, track_height)
        pygame.draw.rect(self.slider_surface, self.slider_color, track_rect)
        pygame.draw.rect(self.slider_surface, self.border_color, track_rect, 1)
        
        # Draw handle
        handle_width = max(8, int(self.w * self.handle_width_fraction))
        handle_height = max(8, int(self.h * self.handle_height_fraction))
        handle_x = self._value_to_position(self._value)
        handle_y = (self.h - handle_height) // 2
        
        self.handle_rect = pygame.Rect(handle_x, handle_y, handle_width, handle_height)
        pygame.draw.rect(self.slider_surface, self.handle_color, self.handle_rect)
        pygame.draw.rect(self.slider_surface, self.border_color, self.handle_rect, 1)
    
    def draw(self, screen: pygame.Surface) -> pygame.Surface:
        """
        Draws the slider component to the screen.
        """        
        if self.slider_surface:
            screen.blit(self.slider_surface, (self.x, self.y))
        
        return screen

    def handle_event(self, event: pygame.event.Event, parent_mouse_pos=None) -> bool:
        """
        Handles events for the slider component.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if the slider value changed (on mouse release), False otherwise.
        """
        super().handle_event(event, parent_mouse_pos)
        self.value_changed = False
        rerender = False
        
        if not self.slider_surface:
            return False  # No surface means not rendered yet
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = self.get_mouse_pos(event, parent_mouse_pos)
                # Check if clicked on slider area
                if (0 <= mouse_pos[0] <= self.w and 0 <= mouse_pos[1] <= self.h):
                    self.is_dragging = True
                    self.mouse_was_pressed_on_slider = True
                    
                    # Calculate drag offset if clicked on handle
                    if self.handle_rect and self.handle_rect.collidepoint(mouse_pos[0], mouse_pos[1]):
                        self.drag_offset = mouse_pos[0] - self.handle_rect.x
                    else:
                        # Clicked elsewhere on track, move handle there
                        self.drag_offset = self.handle_rect.width // 2
                        new_pos = mouse_pos[0] - self.drag_offset
                        self._value = self._position_to_value(new_pos)
                        rerender = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_dragging:  # Left click release
                self.is_dragging = False
                
                # Check if value actually changed since drag started
                if abs(self._value - self.previous_value) > 1e-10:
                    self.value_changed = True
                
                self.previous_value = self._value
                self.mouse_was_pressed_on_slider = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging and self.mouse_was_pressed_on_slider:
                mouse_pos = self.get_mouse_pos(event, parent_mouse_pos)
                
                # Update slider position
                new_pos = mouse_pos[0] - self.drag_offset
                old_value = self._value
                self._value = self._position_to_value(new_pos)
                
                # Only rerender if value actually changed
                if abs(self._value - old_value) > 1e-10:
                    rerender = True
        
        if rerender:
            self.render()
        
        return self.value_changed


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