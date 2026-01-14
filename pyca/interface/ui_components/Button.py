import pygame_gui
from pygame_gui.elements import UIButton
import pygame
from .BaseComponent import BaseComponent

class Button(BaseComponent):
    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(0.05,0.1),font_scale=1.0):
        """
        Initializes the button component.

        Args:
            text (str): The text to display on the button.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the button.
            bg_color (tuple): Background color of the button in RGB format.
            text_color (tuple): Color of the text in RGB format.
            font_path (str): File path to the font to be used.
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        self.text = text
        # Create UIButton from pygame_gui
        self.button = UIButton(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
                               text=self.text,
                               container = self.parent.container if self.parent is not None else None,
                               manager=self.manager)

        self.register_main_component(self.button)


    def render(self):
        """
        Renders the button component with correct positioning and size.
        """
        # Update button position and size
        self.button.set_relative_position((self.x, self.y))
        self.button.set_dimensions((self.w, self.h))

    def handle_event(self, event):
        """
        Handles pygame events for the button.
        Return True if the button was pressed.

        Args:
            event: Pygame event to handle.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.button:
                return True
        
        return False
                