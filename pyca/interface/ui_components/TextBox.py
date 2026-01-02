import pygame_gui
import pygame
from .BaseComponent import BaseComponent


class TextBox(BaseComponent):
    """
    Represents a text box (word-wrapped text) in the UI.
    """
    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.3), font_size=12):
        """
        Initializes the text box component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
            font_size: Base is 12, and changes are proportional based on that
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        self.base_font_size = font_size

        self._text = text
        
        self.textbox = pygame_gui.elements.UITextBox(
            html_text=self._html_text,
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            manager=self.manager,
            container=self.parent.container if self.parent is not None else None
        )

        self.register_main_component(self.textbox)
    
    @property
    def _html_text(self):
        """
        Adjusts the font size in the HTML text based on the current absolute font size.
        """
        print('FONT SIZE:', self.font_abs_size)
        return f"<font face='aldo' size={self.font_abs_size:.1f}>{self._text}</font>"
    
    @property
    def font_abs_size(self):
        """
        Returns the absolute font size based on the base font size and scaling.
        In 0.5 increments, compatible with html font sizing.
        """
        maxi=16
        mini=6
        actual_size = self.BASE_FONT_REL_SIZE * (self.sH+self.sW)/2 * self.base_font_size/28.
        size = (max(mini, min(maxi, actual_size))-mini)/(maxi-mini) # Between 0 and 1

        return int((size*4.6+1)*2)/2.
        
    
    def _adjust_font_size(self):
        self.textbox.set_text(self._html_text)
    
        for child in self.child_components:
            child._adjust_font_size()

        self.textbox.rebuild()


    @property
    def text(self):
        """
        Returns the text of the text box.
        """
        return self._text
    
    @text.setter
    def text(self, new_text):
        """
        Sets the text of the text box 
        """
        self._text = new_text
        self.textbox.set_text(self._html_text)
        self.textbox.rebuild()

    def render(self):
        """
        Renders the text box with correct font size and height/width.
        """
        self.textbox.set_relative_position((self.x, self.y))
        self.textbox.set_dimensions((self.w, self.h))
