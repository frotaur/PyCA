"""
    Class for the main window of PyCA. Deals with the main GUI components,
    main pygame loop, and event handling and dispatching.

    Exposes an API of methods to modify stuff, such as giving the list of active automata,
    running the loop, and default values.
"""
import pygame, os, json
from importlib.resources import files

from pyca.interface import Camera, launch_video, print_screen, add_frame
from .ui_components import SmartFont, TextLabel, DropDown, InputField, InputBox
from ..automata import AUTOMATAS
from .files import DEFAULTS, INTERFACE_HELP, BASE_FONT_PATH

class MainWindow:
    """
    Main window of PyCA application. Deals with the main GUI components,
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


        
        pygame.init()
        self.text_f_size = 1./40
        self.title_f_size = 1./37

        self.font_text = SmartFont(fract_font_size=self.text_f_size, font_path=BASE_FONT_PATH)
        self.font_title = SmartFont(fract_font_size=self.title_f_size, font_path=BASE_FONT_PATH)

        programIcon = pygame.image.load(str(files(f'{__package__}.files') / 'icon.png'))
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("PyCA")



        self.screen = pygame.display.set_mode((self.sW,self.sH), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.camera = Camera(self.sW,self.sH, world_border=(self.W, self.H))
        self.camera.zoom = min(self.sW/self.W, self.sH/self.H)


        # Boolean flags for the main loop
        self.running = True
        self.stopped = True
        self.recording = False
        self.display_help = True
        self.vid_writer=None

        self._initial_automaton = "CA2D"

        self.auto = self.load_automaton(self._initial_automaton)

        # Prep Base GUI elements

        # Text labels for description, help and automaton controls
        self._generate_and_place_left_texts()  

        # FPS live label
        fps_font = SmartFont(fract_font_size=self.text_f_size, font_path=BASE_FONT_PATH, base_sH=self.sH, 
                             max_font_size=16, min_font_size=10)
        self.fps_label = TextLabel(f"FPS: {self.fps}", fract_position=(0.94, 0.01), fract_width=0.3, font=fps_font, color=(230, 120, 120))
        # Dropdown for automaton selection
        drop_size = (0.05, 0.17)  # Fractional size of the dropdown
        drop_pos = (1-drop_size[1]-0.015, 1-drop_size[0]-0.015)  # Position at the bottom right
        self.automaton_dropdown = DropDown(options=list(AUTOMATAS.keys()), fract_position=drop_pos, fract_size=drop_size,open_upward=True)
        self.automaton_dropdown.selected = self._initial_automaton

        # Input boxes for FPS, Width and Height
        fps_size = (0.07, 0.04)
        boxes_size = (0.07, 0.06)
        spacing = 7/1000
        boxes_pos = (1.-fps_size[1]-3*spacing-boxes_size[1]*2, 0.05)

        self.fps_box = InputField(fract_position=(boxes_pos[0], boxes_pos[1]), fract_size=fps_size, label="FPS",
                                 init_text=str(self.fps), allowed_chars=lambda c: c.isdigit(), max_length=3,
                                 font_path=BASE_FONT_PATH)
        self.width_box = InputField(fract_position=(boxes_pos[0]+fps_size[1]+spacing, boxes_pos[1]), fract_size=boxes_size, label="Width",
                                    init_text=str(self.W), allowed_chars=lambda c: c.isdigit(), max_length= 4,
                                    font_path=BASE_FONT_PATH)
        self.height_box = InputField(fract_position=(boxes_pos[0]+fps_size[1]+boxes_size[1]+2*spacing, boxes_pos[1]), fract_size=boxes_size, label="Height",
                                     init_text=str(self.H), allowed_chars=lambda c: c.isdigit(), max_length=4,
                                     font_path=BASE_FONT_PATH)
        
        # Live automaton label
        self.auto_label = TextLabel(text = self.auto.get_string_state(),
                                    fract_position=(0.02, 0.95), fract_width=0.8,color=(180,220,180),
                                    h_margin=0.2, bg_color=(0,0,0,150), font=self.font_text)
        
    def _generate_and_place_left_texts(self):
        """
        Generates and places the left text labels of the main GUI. Needs to do some hacking to get dynamic positions
        of the texts, because a TextLabel component's height cannot be computed before it is rendered (because of text wrapping).
        """
        title_color = (230, 89, 89)
        description_color = (74, 101, 176)
        text_color = (230, 230, 230)
        bg_color = (0,0,0, 150)
        auto_description, auto_help = self.auto.get_help()

        self.auto_name = TextLabel(self.auto.name(), fract_position = (0.005, 0.005), fract_width=0.2, font=self.font_title, color=title_color, bg_color=bg_color, h_margin=0.2)
        self.auto_text = TextLabel(auto_description,fract_position=(0.005,0.), fract_width=0.35,font=self.font_text, color=description_color, bg_color=bg_color)
        self.help_title = TextLabel(INTERFACE_HELP['title'], fract_position=(0.005, 0.05), fract_width=0.2, font=self.font_title, color=title_color, bg_color=bg_color, h_margin=0.2)
        self.help_text = TextLabel(INTERFACE_HELP['content'], fract_position=(0.005, 0.1), fract_width=0.35, font=self.font_text, line_spacing=0.15, color=text_color, bg_color=bg_color)
        self.auto_controls_title = TextLabel("Automaton Controls", fract_position=(0.005, 0.2), fract_width=0.2, font=self.font_title, color=title_color, bg_color=bg_color, h_margin=0.2)
        self.auto_controls_text = TextLabel(auto_help, fract_position=(0.005, 0.25), fract_width=0.35, font=self.font_text, line_spacing=0.15, color=text_color, bg_color=bg_color)

        self.left_components = [
            self.auto_name,
            self.auto_text,
            self.help_title,
            self.help_text,
            self.auto_controls_title,
            self.auto_controls_text
        ]

        fractional_y_position = self.left_components[0].f_pos[1]
        for component in self.left_components:
            component.set_screen_size(self.sH, self.sW)
            component.f_pos = (component.f_pos[0], fractional_y_position)
            component.render()
            fractional_y_position += component.f_size[0] # Add the now correct height to the next component's position
    
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

            self.auto.process_event(event, self.camera)
            self._gui_events(event)
    
    def _gui_events(self,event):
        """
            Handles the base GUI events, for fps, world size and automaton selection.
        """
        if self.automaton_dropdown.handle_event(event):
            selected = self.automaton_dropdown.selected
            self.auto = self.load_automaton(selected)
            self._generate_and_place_left_texts() # Need to update the text labels

        if self.fps_box.handle_event(event):
            try:
                self.fps = int(self.fps_box.value)
            except ValueError:
                print(f"Invalid FPS value: {self.fps_box.value}. Must be a positive integer.")
        if self.width_box.handle_event(event):
            # TODO : find a way for this to be automatic in the automaton (i.e., by default resize and reset)
            # And that way, we can override to do smart resizing
            self.W = int(self.width_box.value)
            print(f"Resizing automaton to width {self.W}")
            self.auto = self.load_automaton(self.automaton_dropdown.selected)  # Reload the automaton with the new width
            self.camera.change_border((self.W, self.H))  # Update the camera border size
        if self.height_box.handle_event(event):
            self.H = int(self.height_box.value)
            self.auto = self.load_automaton(self.automaton_dropdown.selected)  # Reload the automaton with the new height
            self.camera.change_border((self.W, self.H))  # Update the camera border size
    
    def main_loop(self):
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
                self.draw_help()


            pygame.display.flip()
            self.clock.tick(self.fps)
            
        if(self.vid_writer is not None):
            self.vid_writer.release()

        pygame.quit()
    
    def draw_help(self):
        """
            Draws the help text on the screen.
        """
        for component in self.left_components:
            component.draw(self.screen)
        self.automaton_dropdown.draw(self.screen)
        self.fps_label.draw(self.screen)
        self.fps_box.draw(self.screen)
        self.width_box.draw(self.screen)
        self.height_box.draw(self.screen)
        self.auto_label.draw(self.screen)

        self.fps_label.text = f"FPS: {round(self.clock.get_fps())}"
        self.auto_label.text = self.auto.get_string_state()
