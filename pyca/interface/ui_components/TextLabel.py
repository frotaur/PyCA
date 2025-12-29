from .BaseComponent import BaseComponent
import pygame, pygame_gui
from pygame_gui.elements.ui_label import UILabel

class TextLabel(BaseComponent):
    """
    Represents a text label in the UI.
    """

    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.1)):
        """
        Initializes the text label component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
        """

        super().__init__(manager, parent, rel_pos, rel_size)

        self.textlabel = UILabel(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h), manager=self.manager,
                                 text=text,
                                 container = self.parent.container if self.parent is not None else None)
    
    @property
    def text(self):
        """
        Returns the text of the label.
        """
        return self.textlabel.text
    
    @text.setter
    def text(self, new_text):
        """
        Sets the text of the label 
        """
        self.textlabel.set_text(new_text)
    
    def render(self):
        """
        Renders the text with correct font size and height/width.
        """
        self.textlabel.set_relative_position((self.x, self.y))
        self.textlabel.set_dimensions((self.w, self.h))


    def handle_event(self, event:pygame.event.Event) -> bool:
        """
        Text labels do not handle events by default.
        Args:
            event (pygame.event.Event): The event to handle (optional)
        """
        return False
