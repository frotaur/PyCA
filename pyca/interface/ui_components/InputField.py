from .BaseComponent import BaseComponent
from .SmartFont import SmartFont
from .TextLabel import TextLabel
import pygame, time
from ..files import BASE_FONT_PATH
from typing import Callable


class InputField(BaseComponent):
    """
        Basic input field, with an optional label as caption.
        Exact same as InputBox, but with a label on top.
    """
    def __init__(self, fract_position=(0,0), fract_size=(0.1,0.1), label='', init_text='',
                 allowed_chars: Callable | None = None, max_length: int=None,
                 bg_color=(50, 50, 50), text_color=(230, 230, 230), label_color=(230, 230, 230),
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
        # NOTE : since its a BaseComponent composed of sub-componenets, 
        # we need to set the correct screen-fractions for the sub-components
        # Ideally, we wouldn't need to do this (just express as fraction of the mother component),
        # but we can't due to events reporting only screen-relative positions.

        label_fraction_height = 0.3 # Fraction of height the label takes
        label_margins = 0.12
        self.font = SmartFont(font_path=font_path, fract_font_size = fract_size[0]*label_fraction_height, min_font_size=8, max_font_size=18)

        if(label):
            self.label = TextLabel(label, fract_position=fract_position,fract_width=fract_size[1],
                                   font=self.font, bg_color=(0,0,0,150), h_margin=label_margins,
                                   color=label_color)
            input_f_size = (1-label_fraction_height*(1+label_margins), 1.)  # Adjust input size to account for label height
            input_f_pos = (0., label_fraction_height*(1+label_margins))  # Position input below the label
        else:
            self.label = None
            input_f_size = (1., 1.)
            input_f_pos = (0., 0.)
        
        true_f_pos = (fract_position[0] + input_f_pos[0] * fract_size[1],
                      fract_position[1] + input_f_pos[1] * fract_size[0])
        true_f_size = (input_f_size[0] * self.f_size[0], input_f_size[1] * self.f_size[1])

        self.input_box = InputBox(fract_position=true_f_pos, 
                           fract_size=true_f_size, init_text=init_text,
                           allowed_chars=allowed_chars, font_path=font_path,
                           bg_color=bg_color,text_color=text_color)
        
        self.full_surface = None # Will hold both label and input

        self.allowed_chars = allowed_chars
        self.max_length = max_length
    
    
    
    @property
    def value(self):
        """Get the current value of the input box."""
        return self.input_box.value
    
    @value.setter
    def value(self, new_value):
        """Set the value of the input box."""
        self.input_box.value = new_value
    
    def handle_event(self, event):
        """
        Handle events relevant to the input box.
        Returns True if the input was submitted (Enter key, or click away)
        False otherwise.
        """
        return self.input_box._handle_event(event)

    def render(self):
        # Sub-components render themselves
        pass


    def draw(self, screen):
        """
        Draws the input field to the screen, including label if present.
        Args:
            screen (pygame.Surface): The surface to draw on.
        """
        if self.label:
            screen = self.label._draw(screen)
        screen = self.input_box._draw(screen)

        return screen



class InputBox(BaseComponent):
    """
        Basic input box, with configurable allow-list. Its state
        is simply a string of the current text.
    """

    def __init__(self, fract_position=(0,0), fract_size=(0.05,0.1), init_text='',
                 allowed_chars: Callable | None = None, max_length=None, bg_color=(50, 50, 50),
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

        self.font = SmartFont(font_path=font_path, fract_font_size=fract_size[0]*0.4, max_font_size=16, min_font_size=10)

        # State
        self.text = init_text
        self.max_length = max_length
        self.allowed_chars = allowed_chars if allowed_chars is not None else lambda c: True 
        self.previous_text = self.text  # To prevent when not modified

        # Colors
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_color = tuple(min(bg+30,255) for bg in bg_color) 
        self.active_color = tuple(min(bg+20,255) for bg in bg_color)

        # Surfaces
        self.box_surface = None  # Will hold the input box surface
        self.box_rect = None  # Will hold the input box rectangle
        self.border_size_fraction = 0.07  # Border size as a fraction of the dropdown height

    def render(self):
        """
        Renders the input box.
        """
        self.font.sH = self.sH

        self.box_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.box_rect = self.box_surface.get_rect(topleft=(self.x, self.y))
        if self.am_focused():
            bg_color = self.active_color
        else:
            bg_color = self.bg_color

        self.box_surface.fill(bg_color)
        pygame.draw.rect(self.box_surface, self.border_color, (0,0,self.w,self.h), max(1,int(self.h * self.border_size_fraction)))

        text_surface = self.font.render(self.text, self.text_color)
        text_rect = text_surface.get_rect(center=self.box_surface.get_rect().center)
        self.box_surface.blit(text_surface, text_rect)

        return self.box_surface

    @property
    def value(self):
        """
        Get the current value of the input box.
        """
        return self.text

    @value.setter
    def value(self, new_value):
        """
        Set the value of the input box.
        """
        if self.allowed_chars is None or all(self.allowed_chars(c) for c in new_value):
            if self.max_length is None or len(new_value) <= self.max_length:
                self.text = new_value
                self.render()
            else:
                raise ValueError(f"Input exceeds maximum length of {self.max_length}.")
        else:
            raise ValueError("Input contains disallowed characters.")
    
    def draw(self, screen : pygame.Surface):
        """
        Draws the input box on the given surface.
        """
        screen.blit(self.box_surface, (self.x, self.y))

        return screen

    def handle_event(self, event):
        rerender = False
        changed = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state if the user clicked on the input box
            was_active = self.am_focused()
            clicked_on = self.box_rect.collidepoint(event.pos)
            focus_lost = ((not clicked_on) and was_active) # If we just 'input' by clicking away
            if(focus_lost):
                self.release_keyboard_focus()

            if((not was_active) and clicked_on): # Focus gained
                self.text = ""  # Clear text when focused
                self.request_keyboard_focus()
                rerender = True

        if event.type == pygame.KEYDOWN and self.am_focused():        
            if event.key == pygame.K_RETURN:
                self.release_keyboard_focus()
                changed = True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                rerender = True
            else:
                # Add character if it passes the filter and respects the max length
                if self.allowed_chars(event.unicode) and (self.max_length is None or len(self.text) < self.max_length):
                    self.text += event.unicode
                rerender = True
    
        if rerender:
            self.render()

        return changed

    def on_focus_lost(self):
        """
        Called when the input box loses focus.
        """
        self.render()
