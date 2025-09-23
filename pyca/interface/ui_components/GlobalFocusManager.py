import pygame


class GlobalFocusManager:
    def __init__(self):
        self.focused_component = None
        self.mouse_focused_component = None  # Not used for now, maybe later

    def am_focused(self, component):
        return self.focused_component == component

    def request_keyboard_focus(self, component):
        if self.focused_component is not None and self.focused_component != component:
            # FIRST lose focus, then call on_focus_lost
            old_focus = self.focused_component
            self.focused_component = component
            print('FOCUS ON : ', component.__class__.__name__)

            old_focus.on_focus_lost()
        else:
            self.focused_component = component
            print('FOCUS ON : ', component.__class__.__name__)

    
    def release_keyboard_focus(self, component):
        if self.focused_component == component:
            # FIRST lose focus, then call on_focus_lost
            self.focused_component = None
            print('FOCUS LOST : ', component.__class__.__name__)

            component.on_focus_lost()
        
    def should_process_event(self, event, requester=None):
        # NOTE : This is not the best, as also 'automaton' classes ask this question, even though they are
        # not UI components. For now I put requester = None, if not given, will always return False if focused somewhere.
        if event.type in [pygame.KEYDOWN, pygame.KEYUP]: #  If it's a keyboard event
            if(self.focused_component is None):\
                return True  # No component is focused, allow all keyboard events
            else:
                return self.focused_component == requester # Allow only if this component is focused
        else:
            return True  # Non-keyboard events are always processed