import pygame


class GlobalFocusManager:
    def __init__(self):
        self.focused_component = None
        self.mouse_focused_component = None  # Not used for now, maybe later

    def request_keyboard_focus(self, component):
        if self.focused_component is not None and self.focused_component != component:
            self.focused_component.on_focus_lost()

        self.focused_component = component
    
    def release_keyboard_focus(self, component):
        if self.focused_component == component:
            self.focused_component = None
        
    def should_process_event(self, event, requester):
        if event.type in [pygame.KEYDOWN, pygame.KEYUP]: #  If it's a keyboard event
            return self.focused_component == requester # Allow only if this component is focused
        else:
            return True  # Non-keyboard events are always processed