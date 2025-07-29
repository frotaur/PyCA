"""
    Class for the main window of PyCA. Deals with the main GUI components,
    main pygame loop, and event handling and dispatching.

    Exposes an API of methods to modify stuff, such as giving the list of active automata,
    running the loop, and default values.
"""
import pygame, os, json
from importlib.resources import files

from pyca.interface import Camera, launch_video, print_screen, add_frame
from ..automata import AUTOMATAS
from .files import DEFAULTS

class MainWindow:
    """
        Main window of PyCA application.Deals with the main GUI components,
    main pygame loop, and event handling and dispatching.

    Exposes an API of methods to modify stuff, such as giving the list of active automata,
    running the loop, and default values.
    """
    def __init__(self,screen_size=(600, 800), world_size=(200,200), device="cpu"):
        """
            Args:
                screen_size (tuple (H,W)): Size of the screen in pixels.
                world_size (tuple (H,W)): Size of the world in cells.
                device (str): Device to use for the automata. Defaults to "cpu".
        """
        self.sH,self.sW = screen_size
        self.H,self.W = world_size
        self.device = device

        self.fps=60 # Visualization FPS
        self.video_fps=60 # Saved video FPS

        self.font_path = str(files(f'{__package__}.files') / 'AldotheApache.ttf')

        with open(str(files(f'{__package__}.files') / 'std_help.json'), 'r') as f:
            self.std_help = json.load(f)
        
        pygame.init()

        self.font = pygame.font.Font(self.font_path, self.text_size)
        self.font_title = pygame.font.Font(self.font_path, self.title_size)

        programIcon = pygame.image.load(str(files(f'{__package__}.files') / 'icon.png'))
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("PyCA")



        self.screen = pygame.display.set_mode((self.sW,self.sH), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.camera = Camera(self.sW,self.sH, border=(self.W, self.H))
        self.camera.zoom = min(self.sW/self.W, self.sH/self.H)


        # Boolean flags for the main loop
        self.running = True
        self.stopped = True
        self.recording = False
        self.display_help = True
        self.vid_writer=None

        self._initial_automaton = "CA2D"

        self.auto = self.load_automaton(self._initial_automaton)

    def load_automaton(self, automaton_name):
        """
            Loads and returns the specified automaton model.
            Args:
                automaton_name (str): Name of the automaton model to load.
        """
        if automaton_name in AUTOMATAS:
            if(automaton_name in DEFAULTS): 
                defaults = DEFAULTS[automaton_name]
            else:
                defaults = {}
            return AUTOMATAS[automaton_name]((self.H,self.W),**defaults, device=self.device)
        else:
            raise ValueError(f"Invalid automaton model: {automaton_name}. Must be one of {list(AUTOMATAS.keys())}.")

    @property
    def initial_automaton(self):    
        return self._initial_automaton
    
    @initial_automaton.setter
    def initial_automaton(self, value):
        if value in AUTOMATAS:
            self._initial_automaton = value
        else:
            raise ValueError(f"Invalid automaton model: {value}. Must be one of {list(AUTOMATAS.keys())}.")

    @property
    def text_size(self):
        return int(self.sH/45)

    @property
    def title_size(self):
        return int(self.sH/30)

    def update_fonts(self):
        """
            Updates the fonts used in the GUI based on the current screen size.
        """
        self.font = pygame.font.Font(self.font_path, self.text_size)
        self.font_title = pygame.font.Font(self.font_path, self.title_size)

    def handle_events(self):
        """
            Handles all events in the main loop.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            self.camera.handle_event(event)

            if event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_SPACE): # Space to start/stop the automaton
                    self.stopped=not(self.stopped)
                if(event.key == pygame.K_q):
                    self.running=False
                if(event.key == pygame.K_r): # Press 'R' to start/stop recording
                    self.recording = not self.recording
                    if(self.vid_writer is not None and (not self.recording)):
                        # We enter here if we just stopped the recording
                        self.vid_writer.release()
                        self.vid_writer = None
                    elif(self.recording):
                        # We enter here if we just started the recording
                        # TODO : MAKE LAUNCH VIDEO BETTER!!
                        self.vid_writer = launch_video((self.H, self.W), self.video_fps, 'mp4v')
                if(event.key == pygame.K_p):
                    print_screen(self.auto.worldsurface)
                if(event.key == pygame.K_s):
                    self.auto.step()
                if (event.key == pygame.K_h):
                    self.display_help = not self.display_help
                if (event.key == pygame.K_c):
                    self.camera.resize(self.sW,self.sH)
                    self.camera.zoom = min(self.sW/self.W,self.sH/self.H) # Reset zoom to full view
                    self.camera.center()
            
            if event.type == pygame.VIDEORESIZE:
                self.sW, self.sH = event.w, event.h
                self.s_size = (self.sW, self.sH)

                self.camera.resize(self.sW, self.sH)

                self.update_fonts()

            self.auto.process_event(event, self.camera)
            self._gui_events(event)
    
    def _gui_events(self,event):
        """
            Handles the base GUI events, for fps, world size and automaton selection.""q"
        """
        pass # Implement later when we have the GUI elements properly set up

    def game_loop(self):
        """
            Runs the PyCA main loop.
        """

        while self.running:
            self.handle_events()

            if(not self.stopped):
                self.auto.step()

            self.auto.draw()
            world_surface = self.auto.worldsurface
            screen_surface = pygame.Surface((self.sW, self.sH))

            self.screen.fill((0, 0, 0))  # Clear the screen

            # Draw the world surface centered on the screen
            world_rect = world_surface.get_rect()
            world_rect.center = (self.sW // 2, self.sH // 2)
            screen_surface.blit(world_surface, world_rect)

            # Then zoom to current camera view and blit
            screen_surface = self.camera.apply(screen_surface, border=True)
            self.screen.blit(screen_surface, (0, 0))

            if self.recording:
                add_frame(self.vid_writer, world_surface)
                pygame.draw.circle(self.screen, (255, 0, 0), (self.sW - 20, 15), 7)
            
            if(self.display_help):
                pass # Draw here the help text, and all GUI elements

            pygame.display.flip()
            self.clock.tick(self.fps)

        if(self.vid_writer is not None):
            self.vid_writer.release()

        pygame.quit()