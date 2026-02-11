import pygame_gui
import pygame
from .UIComponent import UIComponent


class MultiToggle(UIComponent):
    """
    Represents a multi-toggle button. It is a button that cycles through
    multiple states each time it is pressed.
    """

    def __init__(self, states, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1),state_bg_colors=None, init_state_index=0):
        """
        Initializes the multi-toggle button component.

        Args:
            states (list): List of strings representing the different states.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the button.
            z_pos (int): Z-position for rendering order. Higher values are rendered on top.
            state_bg_colors (list): List of RGB tuples for background colors for each state.
            init_state_index (int): Initial state index.
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        self.states = states
        self.current_state_index = init_state_index


        
        self.button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            text=self.states[self.current_state_index],
            container=self.parent.container if self.parent is not None else None,
            manager=self.manager)
        
        if state_bg_colors is None:
            # Default colors if none provided
            self.state_bg_colors=[self.button.colours["normal_bg"] for _ in range(len(states))] # Do nothing for now, but in the future use default colors somehow
            self.state_bg_active_colors = [self.button.colours["active_bg"] for _ in range(len(states))]
        else:
            if len(state_bg_colors) != len(states):
                raise ValueError("Length of state_bg_colors must match length of states.")
            self.state_bg_colors = state_bg_colors
            self.state_bg_active_colors = [
                        tuple(max(0, bg-30) for bg in color) for color in self.state_bg_colors
                    ]
        self.register_main_component(self.button)
        self._set_aspect()

    def _set_aspect(self):
        """
        Sets the button color based on the current state.
        """
        self.button.set_text(self.states[self.current_state_index])
        self.button.colours["normal_bg"] = pygame.Color(self.state_bg_colors[self.current_state_index])
        # self.button.colours["hovered_bg"] =todo
        self.button.colours["active_bg"] = pygame.Color(self.state_bg_active_colors[self.current_state_index])
        self.button.rebuild()
    
    
    def toggle_state(self):
        """
        Toggles to the next state and updates the button appearance.
        """
        self.current_state_index = (self.current_state_index + 1) % len(self.states)

        self._set_aspect()
    
    @property
    def value(self):
        """
        Returns the current state value.
        """
        return self.states[self.current_state_index]
    
    @value.setter
    def value(self, state):
        """
        Sets the current state to the specified value.

        Args:
            state (str): The state to set. Must be one of the defined states.
        """
        if state not in self.states:
            raise ValueError(f"State '{state}' is not a valid state.")
        self.current_state_index = self.states.index(state)

        self._set_aspect()
    
    @property
    def active(self):
        """
        Returns True if button is in first state, False otherwise.
        Raises ValueError if there are more than 2 states.
        """
        if len(self.states) != 2:
            raise ValueError("Active property is only valid for two-state toggles.")
        return self.current_state_index == 0

    def handle_event(self, event):
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.button:
                self.toggle_state()
                return True
        
        return False
    
    def render(self):
        """
        Renders the multi-toggle button component with correct positioning and size.
        """
        # Update button position and size
        self.button.set_relative_position((self.x, self.y))
        self.button.set_dimensions((self.w, self.h))