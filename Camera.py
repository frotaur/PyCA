import pygame

class Camera:
    """
        Class that handles the 'camera' in a pygame application.
        Allows one to move in the 2D world, zoom in and out by using the mouse + CTRL.
    """
    def __init__(self, width, height):
        """
            Parameters:
            width : int
                Width of the camera
            height : int
                Height of the camera
        """

        self.position = pygame.Vector2(width/2,height/2) # Camera position
        self.zoom = 1.0 # Current zoom value
        self.drag_start = None # Position of the mouse when dragging
        self.size = pygame.Rect(0,0,width,height) # Size of the camera
        self.updateFov() # Update the field of view

    def handle_event(self, event):
        """
            Handles the camera events, such as zooming and dragging. 
            Call it from the main pygame loop, passing the event.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and (pygame.key.get_mods() & pygame.KMOD_CTRL) :
            if event.button == 4 :  # Scroll wheel up
                self.zoom *= 1.1
            elif event.button == 5:  # Scroll wheel down
                self.zoom /= 1.1
            elif event.button == 1:  # Left mouse button
                self.drag_start = pygame.mouse.get_pos()

            if(self.zoom<1):
                self.zoom=1.
            elif(self.zoom>20):
                self.zoom=20

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
            

        self.constrainCam()
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
        """ Takes current mouse position, and converts it in the position given no zoom or camera offset
            
            Params :
            pos : 2-uple (x,y)
                Current mouse position
            
            Returns :
            2-uple (x,y) , absolute position in the world, without zoom or camera offset
        """
        return (pos[0]/self.zoom+self.fov.left, pos[1]/self.zoom+self.fov.top)
    
    def apply(self, surface : pygame.Surface):
        """
            Given a pygame surface, return a new surface which is the view of the camera.

            Parameters:
            surface : pygame.Surface
                Surface to apply the camera to
        """
        visible_surface = pygame.Surface((self.fov.w, self.fov.h))

        visible_surface.blit(surface, (0,0), self.fov)

        return pygame.transform.scale(visible_surface, (self.size.w, self.size.h))
