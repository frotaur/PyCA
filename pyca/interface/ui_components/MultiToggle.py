from ..files import BASE_FONT_PATH
from .Button import Button
import pygame


class MultiToggle(Button):
    def __init__(self, states, fract_position=(0,0), fract_size=(0.05,0.1), init_true=True,
                 state_bg_colors=[(20, 20, 20)]*2,
                 text_color=(230, 230, 230), font_path: str = BASE_FONT_PATH):
        """
        Initializes the toggle component.

        Args:
            states (list): A list of strings representing the text to display in each state.
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the toggle.
            init_true (bool): If True, the initial state is the first state. If False, initial state is the second state.
            state_bg_colors (list): A list of background colors for each state in RGB format (default: dark grey for all).
            text_color (tuple): Color of the text in RGB format.
            font_path (str): File path to the font to be used.
        """
        # Initialize with the first state
        super().__init__(states[0], fract_position, fract_size,
                         bg_color=state_bg_colors[0], text_color=text_color, font_path=font_path)

        # Store toggle-specific attributes
        self.states = states
        self.state_bg_colors = state_bg_colors

        # Current state (index of the active state)
        self.current_state_index = 0

        # Update pressed colors for all states
        self.pressed_colors = [
            tuple(max(0, bg-30) for bg in color) for color in state_bg_colors
        ]

        # Update border colors for all states
        self.border_colors = [
            tuple(min(bg+30, 255) for bg in color) for color in state_bg_colors
        ]
        self.border_colors = [
            tuple(min(bg+30, 255) for bg in color) for color in state_bg_colors
        ]

        self._set_aspect()
    
    def _set_aspect(self):
        """
        Sets the button's appearance based on the current state.
        """
        self.text = self.states[self.current_state_index]
        self.bg_color = self.state_bg_colors[self.current_state_index]
        self.pressed_color = self.pressed_colors[self.current_state_index]
        self.border_color = self.border_colors[self.current_state_index]
        
    def toggle_state(self):
        """
        Toggles between the two states and updates the button appearance.
        """
        self.current_state_index = (self.current_state_index + 1) % len(self.states)
        
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
        return self.states[self.current_state_index]    

    @value.setter
    def value(self, state:str):
        """
        Sets the current state, if state is valid
        """
        if state in self.states:
            self.current_state_index = self.states.index(state)
            self._set_aspect()
        else:
            raise ValueError(f"Invalid state: {state}. Valid states are: {self.states}")
        
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
