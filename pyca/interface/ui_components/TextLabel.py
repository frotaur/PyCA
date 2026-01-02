from .BaseComponent import BaseComponent
import pygame, pygame_gui
from pygame_gui.elements.ui_label import UILabel
from pygame_gui.core import ObjectID
import json
from importlib.resources import files
from .TextBox import TextBox


class TextLabel(BaseComponent):
    """
    Represents a label using TextBox, easier to deal with than UILabel.
    """

    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.1), font_size=12):
        """
        Initializes the text label component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
            font_size: Base is 12, and changes are proportional based on that
        """
        class_id = f"@better_label"
        super().__init__(manager, parent, rel_pos, rel_size,theme_class=class_id)

        self.textbox = TextBox(
            text=text, font_size=font_size, manager=manager, parent=parent, rel_pos=(0,0), rel_size=(-1,1), theme_class=class_id)
        
        self.register_main_component(self.textbox)
    
    @property
    def text(self):
        """
        Returns the text of the text box.
        """
        return self.textbox.text
    
    @text.setter
    def text(self, new_text):
        """
        Sets the text of the text box 
        """
        self.textbox.text = new_text
        self.textbox.rebuild()
