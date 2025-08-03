from .BaseComponent import BaseComponent
from .SmartFont import SmartFont
from .TextLabel import TextLabel
import pygame
from ..files import BASE_FONT_PATH
from typing import Callable


class InputBox(BaseComponent):
    """
        Basic input box, with configurable allow-list. Its state
        is simply a string of the current text.
    """

    def __init__(self, fract_position, fract_size, init_text='',
                 allowed_chars: Callable | None = None, bg_color=(50, 50, 50),
                 text_color=(230, 230, 230),
                 font_path: str=BASE_FONT_PATH) :
        """
        Initializes the input box.
        
        Args:
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the input box.
            text (str): Initial text in the input box.
            allowed_chars (callable|None): Function that takes a character and returns True if it is allowed.
            max_length (int|None): Maximum length of the input text. If None, no limit.
            font_path (str): File path to the font to be used.
        """
        super().__init__(fract_position=fract_position, fract_size=fract_size)
        self.text = init_text

        self.font = SmartFont(font_path=font_path, fract_font_size=fract_size[0]*0.5)

        self.allowed_chars = allowed_chars

        # Colors
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_color = tuple(min(bg+30,255) for bg in bg_color) 
        self.hover_color = tuple(min(bg+20,255) for bg in bg_color)

    def render(self):
        """
        Renders the input box.
        """
        self.font.sH = self.sH

        box_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        box_surface.fill(self.bg_color)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=box_surface.get_rect().center)
        box_surface.blit(text_surface, text_rect)
        return box_surface

class InputField(BaseComponent):
    """
        Basic input field, with configurable allow-list. Its state
        is simply a string of the current text.
    """
    def __init__(self, fract_position, fract_size, label='', init_text='',
                 allowed_chars: Callable | None = None, max_length: int=None,
                 font_path: str=BASE_FONT_PATH) :
        """
        Initializes the input field.
        
        Args:
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the input field (includes label if provided).
            label (str): Optional label for the input field.
            text (str): Initial text in the input field.
            allowed_chars (callable|None): Function that takes a character and returns True if it is allowed.
            max_length (int|None): Maximum length of the input text. If None, no limit.
            font_path (str): File path to the font to be used.
        """
        super().__init__(fract_position=fract_position, fract_size=fract_size)
        self.text = init_text
        label_fraction_height = 0.3 # Fraction of height the label takes

        self.font = SmartFont(font_path=font_path, fract_font_size = label_fraction_height)

        if(label):
            self.label = TextLabel(label, fract_position=(0., 0.),fract_width=1.,
                                   font=self.font)
            self.input_f_size = (1-label_fraction_height, 1.)  # Adjust input size to account for label height
            self.input_f_pos = (0., label_fraction_height)
        else:
            self.label = None
            self.input_f_size = (1., 1.)
            self.input_f_pos = (0., 0.)
        
        self.full_surface = None # Will hold both label and input

        self.allowed_chars = allowed_chars
        self.max_length = max_length
    
    
    
    @property
    def value(self):
        """Get the current value of the input box."""
        return self.text
    
    @value.setter
    def value(self, new_value):
        """Set the value of the input box."""
        self.text = new_value
    
    def handle_event(self, event):
        """
        Handle events relevant to the input box.
        Returns True if the input was submitted (Enter key, or click away)
        False otherwise.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state if the user clicked on the input box
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            return (was_active==True and self.active==False) # Return True if the user clicked away
            
        if event.type == pygame.KEYDOWN and self.active:        
            if event.key == pygame.K_RETURN:
                self.active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # Add character if it passes the filter and respects the max length
                if self.allowed_chars(event.unicode) and (self.max_length is None or len(self.text) < self.max_length):
                    self.text += event.unicode

    def render(self):
        """
        Renders the input field and its label if present.
        """
        self.full_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        if self.label:
            self.full_surface = self.label.draw(self.full_surface)
        

    def draw(self, screen):
        """
        Draws the input field to the screen, including label if present.
        Args:
            screen (pygame.Surface): The surface to draw on.
        """
        super().draw(screen)

        screen.blit(self.full_surface, (self.x, self.y))

        return screen