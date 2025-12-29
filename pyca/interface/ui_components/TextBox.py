import pygame_gui
import pygame
from .BaseComponent import BaseComponent


class TextBox(BaseComponent):
    """
    Represents a text box (word-wrapped text) in the UI.
    """
    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.3)):
        """
        Initializes the text box component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        self.textbox = pygame_gui.elements.UITextBox(
            html_text=text,
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            manager=self.manager,
            container=self.parent.container if self.parent is not None else None
        )

        self.register_main_component(self.textbox)
    
    @property
    def text(self):
        """
        Returns the text of the text box.
        """
        return self.textbox.html_text

    @text.setter
    def text(self, new_text):
        """
        Sets the text of the text box 
        """
        self.textbox.set_text(new_text)
    
    def render(self):
        """
        Renders the text box with correct font size and height/width.
        """
        self.textbox.set_relative_position((self.x, self.y))
        self.textbox.set_dimensions((self.w, self.h))