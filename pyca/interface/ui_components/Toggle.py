from ..files import BASE_FONT_PATH
from .Button import Button
import pygame


class Toggle(Button):
    def __init__(self, state1, state2, fract_position=(0,0), fract_size=(0.05,0.1), init_true=True,
                 state1_bg_color=(20, 80, 20), state2_bg_color=(80, 20, 20),
                 text_color=(230, 230, 230), font_path: str = BASE_FONT_PATH):
        """
        Initializes the toggle component.

        Args:
            state1 (str): The text to display in the first state (initial state).
            state2 (str): The text to display in the second state.
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the toggle.
            init_true (bool): If True, the initial state is state1. If False, initial state is state2.
            state1_bg_color (tuple): Background color for state 1 in RGB format (default: dark green).
            state2_bg_color (tuple): Background color for state 2 in RGB format (default: dark red).
            text_color (tuple): Color of the text in RGB format.
            font_path (str): File path to the font to be used.
        """
        # Initialize with the first state
        super().__init__(state1, fract_position, fract_size, 
                         bg_color=state1_bg_color, text_color=text_color, font_path=font_path)
        
        # Store toggle-specific attributes
        self.state1 = state1
        self.state2 = state2
        self.state1_bg_color = state1_bg_color
        self.state2_bg_color = state2_bg_color
        
        # Current state (True = state1, False = state2)
        self.is_state1 = init_true
        
        # Update pressed colors for both states
        self.state1_pressed_color = tuple(max(0, bg-30) for bg in state1_bg_color)
        self.state2_pressed_color = tuple(max(0, bg-30) for bg in state2_bg_color)
        
        # Update border colors for both states
        self.state1_border_color = tuple(min(bg+30, 255) for bg in state1_bg_color)
        self.state2_border_color = tuple(min(bg+30, 255) for bg in state2_bg_color)

        self._set_aspect()
    
    def _set_aspect(self):
        """
        Sets the button's appearance based on the current state.
        """
        if self.is_state1:
            self.text = self.state1
            self.bg_color = self.state1_bg_color
            self.pressed_color = self.state1_pressed_color
            self.border_color = self.state1_border_color
        else:
            self.text = self.state2
            self.bg_color = self.state2_bg_color
            self.pressed_color = self.state2_pressed_color
            self.border_color = self.state2_border_color
        
    def toggle_state(self):
        """
        Toggles between the two states and updates the button appearance.
        """
        self.is_state1 = not self.is_state1
        
        self._set_aspect()
        
        # Re-render with new state
        self.render()

    @property
    def value(self):
        """
        Returns the current state as a string.
        
        Returns:
            str: The text of the current state.
        """
        return self.state1 if self.is_state1 else self.state2

    @value.setter
    def value(self, state:str):
        """
        Sets the current state, if state is valid
        """
        if state == self.state1:
            self.state = True
        elif state == self.state2:
            self.state = False
        else:
            raise ValueError(f"Invalid state '{state}'. Must be '{self.state1}' or '{self.state2}'.")
        
        self.render()

    @property
    def state(self):
        """
        Returns the current state as a boolean.
        
        Returns:
            bool: True if in state1, False if in state2.
        """
        return self.is_state1

    @state.setter
    def state(self, state1: bool):
        """
        Sets the current state.
        
        Args:
            state1 (bool): True to set to state1, False to set to state2.
        """
        if(self.is_state1 != state1):
            self.toggle_state()
            self.render()
    
    

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handles events for the toggle component. Overrides Button's handle_event
        to add toggle functionality.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if the toggle was just clicked (state changed), False otherwise.
        """
        # Call parent's handle_event to get click detection
        was_clicked = super().handle_event(event)
        
        # If the button was clicked, toggle the state
        if was_clicked:
            self.toggle_state()
        
        return was_clicked
