from ..Automaton import Automaton
import numpy as np
import pygame
from easydict import EasyDict

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

        self.painting = False
        self.erasing = False

        self.m_pos = EasyDict(x=0,y=0)

    @property
    def worldmap(self):
        """
            We redefine the worldmap property because our world is not a torch tensor
        """
        return (255 * self._worldmap.transpose(2,1,0)).astype(dtype=np.uint8)

    def draw(self):
        """
            This function should update the self._worldmap tensor
        """
        self.paint() # Paint in the draw function, since we want it called even when the automaton is paused
        self._worldmap = np.where(self.world[None,...] == 1, self.sand_color[:,None,None], 0.) 

        # Draw a subtle red square around the mouse cursor
        if 0 <= self.m_pos.x < self.w and 0 <= self.m_pos.y < self.h:
            x, y = int(self.m_pos.x), int(self.m_pos.y)
            s = self.brush_size
            self._worldmap[0, max(0,y-s):min(self.h,y+s+1), max(0,x-s):min(self.w,x+s+1)] += .4
    def process_event(self, event, camera):
        """
        LEFT CLICK+DRAG -> add sand
        RIGHT CLICK+DRAG -> remove sand
        MOUSE WHEEL -> resize brush
        """
        m = self.get_mouse_state(camera)

        # Save the mouse state to know when to paint erase
        if m.left:
            self.painting = True
        else:
            self.painting = False
    
        if m.right:
            self.erasing = True
        else:
            self.erasing = False

        # Update the mouse position
        self.m_pos.x = m.x
        self.m_pos.y = m.y

        if event.type == pygame.MOUSEWHEEL:
            # Adjust flow rate with mouse wheel
            self.brush_size = max(1, min(10, self.brush_size + event.y))
    
    def paint(self):
        if(0<=self.m_pos.x < self.w and 0<=self.m_pos.y < self.h):
            if(self.painting):
                # Add interactions when dragging with left-click
                for dx in range(-self.brush_size, self.brush_size+1):
                    for dy in range(-self.brush_size, self.brush_size+1):
                        if np.random.rand() < .5:
                            draw_y, draw_x = self._clamp(int(self.m_pos.y+dy), int(self.m_pos.x+dx))
                            self.world[draw_y,draw_x] = 1
            elif(self.erasing):
                for dx in range(-self.brush_size, self.brush_size+1):
                    for dy in range(-self.brush_size, self.brush_size+1):
                        draw_y, draw_x = self._clamp(int(self.m_pos.y+dy), int(self.m_pos.x+dx))
                        self.world[draw_y,draw_x] = 0
    
    def _clamp(self, y,x):
        """
            Clamps positions such that they are within the 'sandable' world
        """

        return max(0, min(self.h-2, y)), max(1, min(self.w-2, x))
    
    def step(self):
        """
            One timestep of the automaton
        """

        # Then, we simulate the falling of the sand
        for i in range(self.h-2, 0, -1):
            j_mixing = np.random.permutation(np.arange(1,self.w-1)) # Randomize the order of the columns
            for j in j_mixing:
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