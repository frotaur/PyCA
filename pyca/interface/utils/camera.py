import pygame
from math import ceil, floor


class Camera:
    """
        Class that handles the 'camera' in a pygame application.
        Allows one to move in the 2D world, zoom in and out by using the mouse + CTRL.
    """
    def __init__(self, width, height, world_border=None):
        """
            Parameters:
            height : int
                Height of the camera
            width : int
                Width of the camera
            world_border : tuple (optional)
                If given, a centered border will be drawn around a rectangle of size (width,height)
                If given, convert_mouse_pos will by default return the position in the world border rectangle.
        """

        self.position = pygame.Vector2(width/2,height/2) # Camera position
        self._zoom = 1.0 # Current zoom value
        self.drag_start = None # Position of the mouse when dragging
        self.size = pygame.Rect(0,0,width,height) # Size of the camera
        self.updateFov() # Update the field of view

        self.width = width  
        self.height = height

        self.border_size = world_border
        if(self.border_size is not None):
            self.border = pygame.Rect(0,0,world_border[0],world_border[1])
            self.border.center = (width/2,height/2)

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        if(value<1.):
            self._zoom = 1.
        elif(value>20.):
            self._zoom = 20.
        else:
            self._zoom = value

    def resize(self, width, height):
        """
            Resize the camera
        """
        self.size = pygame.Rect(0,0,width, height)
        rescale_vector = (width/self.width, height/self.height)
        self.width = width
        self.height = height
        self.position = pygame.Vector2(rescale_vector[0]*self.position.x, rescale_vector[1]*self.position.y)
        if(self.border_size is not None):
            self.border.center = (width/2,height/2)
        self.updateFov()

    def change_border(self, new_border):
        """
            Change the border size of the camera.
            If None, no border will be drawn.
        """
        if(new_border is None):
            self.border_size = None
            self.border = None
        else:
            self.border_size = new_border
            self.border = pygame.Rect(0,0,new_border[0],new_border[1])
            self.border.center = (self.size.w/2,self.size.h/2)
        self.updateFov()

    def center(self):
        self.position = pygame.Vector2(self.width / 2, self.height / 2)

    def handle_event(self, event, constrain=False):
        """
            Handles the camera events, such as zooming and dragging. 
            Call it from the main pygame loop, passing the event.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            if event.button == 4 and (pygame.key.get_mods() & pygame.KMOD_CTRL):  # Scroll wheel up
                old_zoom = self.zoom
                self.zoom *= 1.1
                if(self.zoom!= old_zoom):
                    # Adjust position to keep mouse point fixed
                    mouse_pos = pygame.mouse.get_pos()
                    self.position.x += (mouse_pos[0] - self.size.w/2) * (1/old_zoom - 1/self.zoom)
                    self.position.y += (mouse_pos[1] - self.size.h/2) * (1/old_zoom - 1/self.zoom)
            elif event.button == 5 and (pygame.key.get_mods() & pygame.KMOD_CTRL):  # Scroll wheel down
                old_zoom = self.zoom
                self.zoom /= 1.1

                if(self.zoom!= old_zoom):
                    # Adjust position to keep mouse point fixed
                    mouse_pos = pygame.mouse.get_pos()
                    self.position.x += (mouse_pos[0] - self.size.w/2) * (1/old_zoom - 1/self.zoom)
                    self.position.y += (mouse_pos[1] - self.size.h/2) * (1/old_zoom - 1/self.zoom)
            elif event.button == 1:  # Left mouse button
                self.drag_start = pygame.mouse.get_pos()


        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.drag_start = None
        elif event.type == pygame.MOUSEMOTION:
            if self.drag_start is not None:
                x, y = event.pos
                x0, y0 = self.drag_start
                self.position.x += (x0 - x) / self.zoom
                self.position.y += (y0 - y) / self.zoom
                self.drag_start = event.pos
            

        if constrain: self.constrainCam()
        self.updateFov()

    def constrainCam(self):
        """
            Constrain the camera position according to the size of the camera and window
        """
        fsize = pygame.Vector2(self.fov.size)/2
        minx,miny = fsize
        maxx,maxy = pygame.Vector2(self.size.size)-fsize

        self.position.x = max(min(self.position.x,maxx),minx)
        self.position.y = max(min(self.position.y,maxy),miny)

    def updateFov(self):
        """
            Updates field of view.
        """
        self.fov=pygame.Rect(0,0,int(self.size.w/self.zoom),int(self.size.h/self.zoom))
        self.fov.center = self.position
    
    def convert_mouse_pos(self, pos):
        """ Takes current mouse position on screen, and converts it in the absolution position,
         given no zoom or camera offset. If world border is set, it will return values inside the 
         world border, and clamped.
            
            Params :
            pos : 2-uple (x,y)
                Current mouse position
            
            Returns :
            2-uple (x,y) , absolute position in the world, without zoom or camera offset
        """
        abs_pos = (int(pos[0]/self.zoom+self.fov.left), int(pos[1]/self.zoom+self.fov.top)) # Position in the screen

        if self.border is not None:
            world_pos = (abs_pos[0] - self.border.left, abs_pos[1] - self.border.top)
            world_pos = (max(0, min(world_pos[0], self.border.width)),
                       max(0, min(world_pos[1], self.border.height)))

            return world_pos

        return abs_pos

    def mouse_in_border(self, pos):
        """
            Returns True if the mouse is inside the world border, if set.
            If no border is set, always returns True.
        """
        if(self.border is None):
            return True
        abs_pos = (int(pos[0]/self.zoom+self.fov.left), int(pos[1]/self.zoom+self.fov.top)) # Position in the screen
        world_pos = (abs_pos[0] - self.border.left, abs_pos[1] - self.border.top)
        if(world_pos[0]<0 or world_pos[0]>=self.border.width or world_pos[1]<0 or world_pos[1]>=self.border.height):
            return False
        
        return True
    
    def apply(self, surface : pygame.Surface, border=False):
        """
            Given a pygame surface, return a new surface which is the view of the camera.

            Parameters:
            surface : pygame.Surface
                Surface to apply the camera to
            border : bool
                If True, a border will be drawn (if border is not None)
        """
        visible_surface = pygame.Surface((self.fov.w, self.fov.h))

        visible_surface.blit(surface, (0,0), self.fov)
        scaled_surface = pygame.transform.scale(visible_surface, (self.size.w, self.size.h))
        if border:
            self.draw_border(scaled_surface)
        return scaled_surface
    
    def draw_border(self, surface: pygame.Surface, thickness: int = 1, color: tuple = (125, 125, 125)):
        """
        Draw a border around the simulation area on the given surface.
        
        Parameters:
        surface : pygame.Surface
            Surface to draw the border on
        thickness : int
            Border thickness in pixels
        color : tuple
            RGB color tuple for the border
        """
        if(self.border is None):
            return
        # NOTE Ceils, and offset values are obtained empirically, that's the best
        # way I found to distribute rounding errors the same left and right, top and bottom.
        # Actual sizes are with no ceil, and with -1 on left and top, and +2 on width and height
        
        # Convert world coordinates to screen coordinates
        left = ceil((-self.fov.left  +self.border.left-.75) * self.zoom)
        top = ceil((-self.fov.top  +self.border.top-.75) * self.zoom)

        # Calculate the scaled width and height
        width = ceil((self.border.width+2) * self.zoom)
        height = ceil((self.border.height+2) * self.zoom)

        # Draw the border
        pygame.draw.rect(surface, 
                        color,
                        (left, top, width, height),
                        int(thickness*self.zoom))