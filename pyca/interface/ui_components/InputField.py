from .BaseComponent import BaseComponent
import pygame_gui
from pygame_gui.elements import UITextEntryLine
import pygame


class InputField(BaseComponent):
    """
        Basic input field, with configurable allow-list. Its state
        is simply a string of the current text.
    """

    def __init__(self, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1), init_text='',
                 allowed_chars: list | None = None, forbidden_chars: list | None = None, max_length: int=None):
        """
        Initializes the input box.
        
        Args:
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the input box
            init_text (str): Initial text in the input box.
            allowed_chars (list|None): List of allowed characters. If None, all characters are allowed.
            forbidden_chars (list|None): List of forbidden characters. If None, no characters are forbidden.
            max_length (int|None): Maximum length of the input text. If None, no limit.
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        self.text_entry = UITextEntryLine(relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
                                          container = self.parent.container if self.parent is not None else None,
                                          manager=self.manager, initial_text=init_text)
        if(max_length is not None):
            self.text_entry.set_text_length_limit(max_length)
        
        if(allowed_chars is not None):
            self.text_entry.set_allowed_characters(allowed_chars)
        if(forbidden_chars is not None):
            if(allowed_chars is not None):
                raise ValueError("Cannot set both allowed_chars and forbidden_chars.")
            self.text_entry.set_forbidden_characters(forbidden_chars)
        
    def render(self):
        """
        Renders the input box.
        """
        self.text_entry.set_relative_position((self.x, self.y))
        self.text_entry.set_dimensions((self.w, self.h))

    @property
    def value(self):
        """
        Get the current value of the input box.
        """
        return self.text_entry.get_text()

    @value.setter
    def value(self, new_value):
        """
        Set the value of the input box.
        """
        self.text_entry.set_text(new_value)


    def handle_event(self, event):
        """
            Returns True if the input was submitted (Enter key, or click away)
        """
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element == self.text_entry:
                return True


