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

        # Define constants for elements
        self.EMPTY = 0
        self.SAND = 1
        self.WATER = 2
        
        self.element = self.SAND  # Start with sand
        self.lowshift_map = {
            self.SAND: 1,   # Sand only moves diagonally down
            self.WATER: 0   # Water can move horizontally
        }
        self.spread_speed = {
            self.SAND: 1,   # Sand spreads 1 cell at a time
            self.WATER: 3   # Water spreads up to 3 cells at a time
        }
        self.lowshift = self.lowshift_map[self.element]
        
        self.world = np.zeros((self.h, self.w))
        self.world[50,50] = self.SAND
        self.sand_color = np.array([0.961, 0.8, 0.208])
        self.water_color = np.array([0, 145, 156])/255
        
        self.left_pressed = False
        self.right_pressed = False
        self.brush_size = 4
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
        
        # Create a mask for each element type
        sand_mask = self.world[None, ...] == self.SAND
        water_mask = self.world[None, ...] == self.WATER
        
        # Apply colors using the masks
        self._worldmap = np.where(sand_mask, self.sand_color[:,None,None],
                                np.where(water_mask, self.water_color[:, None, None], 0.))
        # Draw a subtle red square around the mouse cursor
        if 0 <= self.m_pos.x < self.w and 0 <= self.m_pos.y < self.h:
            x, y = int(self.m_pos.x), int(self.m_pos.y)
            s = self.brush_size
            self._worldmap[0, max(0,y-s):min(self.h,y+s+1), max(0,x-s):min(self.w,x+s+1)] += .3
    
    def process_event(self, event, camera):
        """
        LEFT CLICK -> add sand
        RIGHT CLICK -> remove sand
        MOUSE WHEEL -> adjust flow rate
        E -> toggle sand/water
        DEL -> clear the world
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

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                # Toggle between sand and water
                self.element = self.WATER if self.element == self.SAND else self.SAND
                self.lowshift = self.lowshift_map[self.element]
            if event.key == pygame.K_DELETE:
                # Clear the world
                self.world = np.zeros((self.h, self.w))

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
                            self.world[draw_y,draw_x] = self.element
            elif(self.erasing):
                for dx in range(-self.brush_size, self.brush_size+1):
                    for dy in range(-self.brush_size, self.brush_size+1):
                        draw_y, draw_x = self._clamp(int(self.m_pos.y+dy), int(self.m_pos.x+dx))
                        self.world[draw_y,draw_x] = self.EMPTY
    
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
                if(self.world[i,j] != self.EMPTY):
                    el = self.world[i,j]
                    ls = self.lowshift_map[el]
                    spread = self.spread_speed[el]
                    
                    below = self.world[i+1,j] == self.EMPTY
                    # Check for empty spaces up to spread distance in both directions
                    left_spaces = 0
                    right_spaces = 0
                    
                    for k in range(1, spread + 1):
                        if j-k >= 0 and self.world[i+ls,j-k] == self.EMPTY:
                            left_spaces = k
                        else:
                            break
                            
                    for k in range(1, spread + 1):
                        if j+k < self.w and self.world[i+ls,j+k] == self.EMPTY:
                            right_spaces = k
                        else:
                            break
                    
                    if below:  # If we can fall
                        self.world[i+1,j] = el
                        self.world[i,j] = self.EMPTY
                    elif left_spaces > 0 and right_spaces > 0:  # Can spread both ways
                        if np.random.rand() < 0.5:
                            self.world[i+ls,j-left_spaces] = el
                        else:
                            self.world[i+ls,j+right_spaces] = el
                        self.world[i,j] = self.EMPTY
                    elif right_spaces > 0:  # Can spread right
                        self.world[i+ls,j+right_spaces] = el
                        self.world[i,j] = self.EMPTY
                    elif left_spaces > 0:  # Can spread left
                        self.world[i+ls,j-left_spaces] = el
                        self.world[i,j] = self.EMPTY
                    
                    if self.world[i,j] == self.SAND:
                        # Check if we can swap with sand
                        if self.world[i+1,j] == self.WATER:
                            self.world[i,j], self.world[i+1,j] = self.world[i+1,j], self.world[i,j]
