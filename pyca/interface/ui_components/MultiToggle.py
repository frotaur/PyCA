import pygame_gui
import pygame
from .BaseComponent import BaseComponent


class MultiToggle(BaseComponent):
    """
    Represents a multi-toggle button. It is a button that cycles through
    multiple states each time it is pressed.
    """

    def __init__(self, states, manager, parent=None, rel_pos=(0,0), rel_size=(0.1,0.1),
                 state_bg_colors=None, init_state_index=0):
        """
        Initializes the multi-toggle button component.

        Args:
            states (list): List of strings representing the different states.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the button.
            state_bg_colors (list): List of RGB tuples for background colors for each state.
            init_state_index (int): Initial state index.
        """
        super().__init__(manager, parent, rel_pos, rel_size)

        self.states = states
        self.current_state_index = init_state_index

        if state_bg_colors is None:
            # Default colors if none provided
            self.state_bg_colors = [(200, 200, 200) for _ in states]
        else:
            if len(state_bg_colors) != len(states):
                raise ValueError("Length of state_bg_colors must match length of states.")
            self.state_bg_colors = state_bg_colors

        self.state_bg_active_colors = [
                    tuple(max(0, bg-30) for bg in color) for color in self.state_bg_colors
                ]
        
        self.button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(self.x, self.y, self.w, self.h),
            text=self.states[self.current_state_index],
            container=self.parent.container if self.parent is not None else None,
            manager=self.manager)
        
        self._set_aspect()
        self.register_main_component(self.button)

    def _set_aspect(self):
        """
        Sets the button color based on the current state.
        """
        self.button.set_text(self.states[self.current_state_index])
        self.button.colours["normal_bg"] = self.state_bg_colors[self.current_state_index]
        # self.button.colours["hovered_bg"] = col
        self.button.colours["active_bg"] = self.state_bg_active_colors[self.current_state_index]
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
    
    def handle_event(self, event):
        
        if self.button.handle_event(event):
            self.toggle_state()
            return True