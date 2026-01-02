from .BaseComponent import BaseComponent
import pygame, pygame_gui
from pygame_gui.elements.ui_label import UILabel
from pygame_gui.core import ObjectID
import json
from importlib.resources import files

class TextLabel(BaseComponent):
    """
    Represents a text label (single line) in the UI.
    """

    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.1), font_size='normal'):
        """
        Initializes the text label component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
            font_size: 'big', 'normal' or 'small' for different font sizes. May support arbitrary scaling in the future.
        """

        super().__init__(manager, parent, rel_pos, rel_size)

        id_str = f"@label_{font_size}"
        self.textlabel = UILabel(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h), manager=self.manager,
                                 text=text,
                                 container = self.parent.container if self.parent is not None else None,
                                 object_id=ObjectID(class_id=id_str))
        # NOTE: for now, sizing with themes is bugged, I use a workaround
        theme_workaround_file = files("pyca.interface.ui_components.styling").joinpath("theme_workaround.json")
        with open(theme_workaround_file, "r") as f:
            size_dict = json.load(f)

        self.base_font_size = size_dict[id_str]["font_size"]
        self.register_main_component(self.textlabel)

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
        self.main_element.rebuild()

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
