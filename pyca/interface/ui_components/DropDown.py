from .BaseComponent import BaseComponent
import pygame_gui
from pygame_gui.elements import UIDropDownMenu
import pygame


class DropDown(BaseComponent):
    """
        Component for creating a dropdown/dropup menu, with text options to select from.
        Its state (self.selected) is simply a string. 
    """

    def __init__(self, options, manager, parent=None, rel_pos=(0.,0.), rel_size=(0.05,0.1), open_upward=False, starting_option=None):
        """
        Initializes the dropdown component.

        Args:
            options (list): List of string options of the dropdown.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the dropdown WHEN CLOSED.
            open_upward (bool): If True, dropdown opens upward; if False, opens downward.
        """
        super().__init__(manager, parent, rel_pos, rel_size)
        self.options = options
        self.exp_direction = 'up' if open_upward else 'down'
        self.selected = options[0] if starting_option is None else starting_option
        # Create UIDropDownMenu from pygame_gui
        self.dropdown = UIDropDownMenu(options_list=self.options,
                                       starting_option=self.selected,
                                       relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
                                       container = self.parent.container if self.parent is not None else None,
                                       manager=self.manager,
                                       expand_on_option_click=True)
        
        self.dropdown.expand_direction = self.exp_direction
        for state_name, state in self.dropdown.menu_states.items():
            state.expand_direction = self.exp_direction
        self.dropdown.rebuild()
        self.register_main_component(self.dropdown)
        
    def handle_event(self, event):
        """
        Handles pygame events for the dropdown.
        Updates the selected option if changed.
        Return True if the selection was changed.

        Args:
            event: Pygame event to handle.
        """
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.dropdown:
                self.selected = event.text
                return True
        return False

    def render(self):
        """
        Renders the dropdown component with correct positioning and size.
        """
        # Update dropdown position and size
        self.dropdown.set_relative_position((self.x, self.y))
        self.dropdown.set_dimensions((self.w, self.h))
