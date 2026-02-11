import pygame_gui
import pygame
from .UIComponent import UIComponent


class TextBox(UIComponent):
    """
    Represents a text box (word-wrapped HTML text) in the UI.
    """
    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.3),*, font_size=12,font_color=(230,230,230),text_align='left',selectable=False,theme_class=None,theme_id=None):
        """
        Initializes the text box component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
            font_size: Base is 12, and changes are proportional based on that
            font_color (tuple): Color of the text in RGB format.
            text_align (str): Text alignment within the box. Options are 'left', 'center', 'right'.
            selectable (bool): Whether the text is selectable.
        """
        super().__init__(manager, parent, rel_pos, rel_size, theme_class=theme_class,theme_id=theme_id)

        self.base_font_size = font_size

        self._text = text
        self.font_color = font_color

        self.textbox = pygame_gui.elements.UITextBox(
            html_text=self._html_text,
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            manager=self.manager,
            container=self.parent.container if self.parent is not None else None,
            object_id=self.object_id
        )
        self.textbox.text_horiz_alignment = text_align

        if(not selectable):
            self.textbox.disable()
    
        self.register_main_component(self.textbox)

    @property
    def _html_text(self):
        """
        Adjusts the font size in the HTML text based on the current absolute font size.
        """
        return f"<font face='aldo' color={self.to_hex(self.font_color)} pixel_size={self.font_abs_size}>{self._text}</font>"
    
    @property
    def font_abs_size(self):
        """
        Returns the absolute font size based on the base font size and scaling.
        In 0.5 increments, compatible with html font sizing.
        """

        actual_size = self.BASE_FONT_REL_SIZE * (self.sH+self.sW)/2 * self.base_font_size/12.
        # size = (max(mini, min(maxi, actual_size))-mini)/(maxi-mini) # Between 0 and 1

        return int(actual_size)
        
    
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

    def to_hex(self, rgb):
        r, g, b = rgb
        return f'#{r:02x}{g:02x}{b:02x}'