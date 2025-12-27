from ..files import BASE_FONT_PATH
from .BaseComponent import BaseComponent
from .SmartFont import SmartFont
import pygame


class Button(BaseComponent):
    def __init__(self, text, fract_position=(0,0), fract_size=(0.05,0.1), 
                 bg_color=(50, 50, 50), text_color=(230, 230, 230), font_path: str=BASE_FONT_PATH):
        """
        Initializes the button component.

        Args:
            text (str): The text to display on the button.
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the button.
            bg_color (tuple): Background color of the button in RGB format.
            text_color (tuple): Color of the text in RGB format.
            font_path (str): File path to the font to be used.
        """
        super().__init__(fract_position, fract_size)
        self.text = text
        self.bg_color = bg_color
        self.text_color = text_color
        self.font = SmartFont(fract_font_size=self.f_size[0]*0.4, font_path=font_path)

        # Colors for different states
        self.pressed_color = tuple(max(0, bg-30) for bg in bg_color)  # Darker when pressed
        self.border_color = tuple(min(bg+30, 255) for bg in bg_color)  # Lighter border
        self.border_size_fraction = 0.05  # Border size as fraction of button height

        # State
        self.is_pressed = False
        self.just_clicked = False
        self.mouse_was_pressed_on_button = False

        # Rendered surfaces
        self.button_surface = None
        self.button_rect = None

    def render(self):
        """
        Renders the button component with correct font size and positioning.
        """
        self.font.sH = self.sH
        
        # Choose background color based on state
        bg_color = self.pressed_color if self.is_pressed else self.bg_color
        
        # Create button surface
        self.button_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.button_surface.fill(bg_color)
        
        # Draw border
        pygame.draw.rect(self.button_surface, self.border_color, 
                        (0, 0, self.w, self.h), max(1, int(self.border_size_fraction * self.h)))
        
        # Render text
        text_surface = self.font.render(self.text, self.text_color)
        text_rect = text_surface.get_rect()
        text_rect.center = (self.w / 2, self.h / 2)
        self.button_surface.blit(text_surface, dest=text_rect)
        
        # Update button rectangle
        self.button_rect = pygame.Rect(self.x, self.y, self.w, self.h)

    def draw(self, screen: pygame.Surface) -> pygame.Surface:
        """
        Draws the button component to the screen.
        """
        
        # Draw button
        if self.button_surface:
            screen.blit(self.button_surface, (self.x, self.y))
        
        return screen

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handles events for the button component.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if the button was just clicked, False otherwise.
        """
        self.just_clicked = False
        rerender = False
        
        if not self.button_surface:
            return False  # No surface means not rendered yet
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                if self.button_rect.collidepoint(mouse_pos):
                    self.is_pressed = True
                    self.mouse_was_pressed_on_button = True
                    rerender = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                mouse_pos = event.pos
                old_pressed = self.is_pressed
                self.is_pressed = False
                
                # Check if we have a complete click (pressed on button, released on button)
                if (self.mouse_was_pressed_on_button and 
                    self.button_rect.collidepoint(mouse_pos)):
                    self.just_clicked = True
                
                self.mouse_was_pressed_on_button = False
                
                if old_pressed:  # Only rerender if state actually changed
                    rerender = True
        
        if rerender:
            self.render()
        
        return self.just_clicked