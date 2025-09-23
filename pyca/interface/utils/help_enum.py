


class HelpEnum():
    """
        Has three states: 'NONE', 'SOME', 'ALL'.
        Use toggle() to switch between them.
    """
    def __init__(self):
        
        self.state_dict = {0:'ALL', 1:'SOME', 2:'NONE'}
        self._state = 0

    def toggle(self):
        self._state = (self._state + 1) % 3
    
    @property
    def left_pane(self):
        return self.state in ['ALL']

    @property
    def right_pane(self):
        return self.state in ['ALL', 'SOME']
    

    @property
    def state(self):
        return self.state_dict[self._state]