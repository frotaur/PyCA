from ..Automaton import Automaton
import numpy as np
import pygame


class FallingSand(Automaton):
    """
        Inefficient (sequential) implementation of the falling sand automaton.
    """

    def __init__(self, size):
        super().__init__(size)
        self.h, self.w = size
        self.size = size
        
        self.world = np.zeros((self.h, self.w))
        self.world[50,50] = 1
        self.sand_color = np.array([0.961, 0.8, 0.208])
        
        self.left_pressed = False
        self.right_pressed = False
        self.brush_size = 2
        self.flow_rate = 1  # Number of sand particles to add per click/drag

    @property
    def worldmap(self):
        """
            Converts _worldmap to a numpy array, and returns it in a pygame-plottable format (H,W,3).
        """
        return (255 * self._worldmap.transpose(1,0,2)).astype(dtype=np.uint8)

    def draw(self):
        """
            This function should update the self._worldmap tensor
        """
        self._worldmap = np.where(self.world[..., None] == 1, self.sand_color[None,None,:], 0.) 
    
    def process_event(self, event, camera):
        """
        LEFT CLICK -> add sand
        RIGHT CLICK -> remove sand
        MOUSE WHEEL -> adjust flow rate
        """
        if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
            # Get mouse position and convert to world coordinates
            mouse_pos = pygame.mouse.get_pos()
            x, y = camera.convert_mouse_pos(mouse_pos)
            
            # Add bounds checking
            h, w = self.world.shape
            x = int(x)
            y = int(y)
            
            if 0 <= x < w and 0 <= y < h:  # Only modify if within bounds
                # Add multiple sand particles based on flow rate
                for dx in range(-self.brush_size, self.brush_size+1):
                    for dy in range(-self.brush_size, self.brush_size+1):
                        new_x = x + dx
                        new_y = y + dy
                        if 0 <= new_x < w and 0 <= new_y < h:
                            if np.random.rand() < self.flow_rate/4:  # Randomize particle placement
                                self.world[new_y, new_x] = 1

        if event.type == pygame.MOUSEBUTTONUP:
            if(event.button==1):
                self.left_pressed=False
            elif(event.button==3):
                self.right_pressed=False
        if event.type == pygame.MOUSEMOTION:
            if(self.left_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                # Add interactions when dragging with left-click
                for dx in range(-self.brush_size, self.brush_size+1):
                    for dy in range(-self.brush_size, self.brush_size+1):
                        if np.random.rand() < self.flow_rate/4:
                            self.world[int(y+dy),int(x+dx)] = 1
            elif(self.right_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                # Add interactions when dragging with right-click
                self.world[int(y),int(x)] = 0

        if event.type == pygame.MOUSEWHEEL:
            # Adjust flow rate with mouse wheel
            self.flow_rate = max(0.25, min(10, self.flow_rate + event.y * 0.25))
            print(f"Flow rate: {self.flow_rate:.2f}", end='\r')
            
    
    def step(self):
        """
            One timestep of the automaton
        """
        for i in range(self.h-2, 0, -1):
            for j in range(1, self.w-1):
                if(self.world[i,j] == 1):
                    if self.world[i+1,j] == 0: # If we can fall
                        self.world[i+1,j] = 1
                        self.world[i,j] = 0
                    elif self.world[i+1,j-1] == 0 or self.world[i+1,j+1] == 0:
                        left_free = self.world[i+1,j-1] == 0
                        right_free = self.world[i+1,j+1] == 0
                        if(left_free and right_free):
                            if np.random.rand() < 0.5:
                                self.world[i+1,j-1] = 1
                            else:
                                self.world[i+1,j+1] = 1
                            self.world[i,j] = 0
                        elif left_free:
                            self.world[i+1,j-1] = 1
                            self.world[i,j] = 0
                        elif right_free:
                            self.world[i+1,j+1] = 1
                            self.world[i,j] = 0        