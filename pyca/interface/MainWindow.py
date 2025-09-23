"""
    Class for the main window of PyCA. Deals with the main GUI components,
    main pygame loop, and event handling and dispatching.

    Exposes an API of methods to modify stuff, such as giving the list of active automata,
    running the loop, and default values.
"""
import pygame
from importlib.resources import files

from pyca.interface import Camera, launch_video, print_screen, add_frame
from .ui_components import BaseComponent,SmartFont, TextLabel, DropDown, InputField, Button, MultiToggle, Toggle
from .utils.help_enum import HelpEnum
from ..automata import AUTOMATAS
from .files import DEFAULTS, INTERFACE_HELP, BASE_FONT_PATH

class MainWindow:
    """
    Main window of PyCA application. Deals with the main GUI components,
    main pygame loop, and event handling and dispatching.

    Exposes an API of methods to modify stuff, such as giving the list of active automata,
    running the loop, and default values.
    """
    def __init__(self,screen_size=(600, 800), world_size=(200,200), device="cpu", tablet_mode=False):
        """
            Args:
                screen_size (tuple (H,W)): Size of the screen in pixels.
                world_size (tuple (H,W)): Size of the world in cells.
                device (str): Device to use for the automata. Defaults to "cpu".
                table_mode (bool): If true, will add a clickable UI for the basic functionality
        """
        self.sH,self.sW = screen_size
        self.H,self.W = world_size
        self.device = device

        self.fps=60 # Visualization FPS
        self.video_fps=60 # Saved video FPS
        self.tablet_mode = tablet_mode

        pygame.init()
        self.text_f_size = 1./40
        self.title_f_size = 1./37

        self.font_text = SmartFont(fract_font_size=self.text_f_size, font_path=BASE_FONT_PATH)
        self.font_title = SmartFont(fract_font_size=self.title_f_size, font_path=BASE_FONT_PATH)

        programIcon = pygame.image.load(str(files(f'{__package__}.files') / 'icon.png'))
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("PyCA")


        if self.tablet_mode:
            flags = pygame.RESIZABLE | pygame.FULLSCREEN
        else:
            flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.sW,self.sH), flags)
        self.clock = pygame.time.Clock()
        self.camera = Camera(self.sW,self.sH, world_border=(self.W, self.H))
        self.camera.zoom = min(self.sW/self.W, self.sH/self.H)


        # Boolean flags for the main loop
        self.running = True
        self.stopped = True
        self.recording = False
        self.display_help = HelpEnum() # 'ALL' by default
        self.vid_writer=None

        self._initial_automaton = "MaCELenia"

        ## Right Panel Base Component
        # Define a position where we can put extra components. Its moved appropriately as we add stuff
        self.right_components = None
        self.extra_components_pos = (0,0) 
        self._generate_right_base_gui()
        if(self.tablet_mode):
            self.tablet_gui_components = None
            self._generate_tablet_gui(start_position=self.extra_components_pos)
        self._generate_auto_controls_title(start_position=self.extra_components_pos)

        # Load the initial automaton
        self.auto = self.load_automaton(self._initial_automaton)

        # Text labels for description, help and automaton controls
        self.left_components = None
        self._generate_and_place_left_texts()  

        
    def _generate_tablet_gui(self, start_position=(0.8,0.1)):
        """
            Generates the tablet-mode GUI components, which are buttons for play/pause,
            step, reset, and a dropdown for automaton selection.
        """
        H_SPACING = 0.007
        W_SPACING = 0.01
        BUTTONS_SIZE = (0.05,0.05)
        next_pos = start_position
        self.play_pause = Toggle(state1="Run", state2="Stop", fract_position=next_pos, fract_size=BUTTONS_SIZE)
        next_pos = (next_pos[0]+BUTTONS_SIZE[1]+W_SPACING, next_pos[1])
        self.step = Button(text="Step", fract_position=next_pos, fract_size=BUTTONS_SIZE)
        next_pos = (next_pos[0]+BUTTONS_SIZE[1]+W_SPACING, next_pos[1])
        self.hide_show = MultiToggle(states=["Hide Some", "Hide All", "Show"], fract_position=next_pos, fract_size=BUTTONS_SIZE,
                       state_bg_colors=[(100, 100, 100), (20, 20, 80), (80, 20, 20)])
        # For now, cam reset not needed as we can't move the camera in tablet mode
        # self.cam_reset = Button(text="Center", fract_position=next_pos, fract_size=BUTTONS_SIZE)
        next_pos = (start_position[0], next_pos[1]+BUTTONS_SIZE[0]+H_SPACING)
        self.tablet_gui_components = [
            self.play_pause,
            self.step,
            self.hide_show
        ]

        self.extra_components_pos = (next_pos[0], next_pos[1])
    
    def _generate_right_base_gui(self):
        """
            Generates the base GUI which go on the right side of the window
        """
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

        self.extra_components_pos = (boxes_pos[0], boxes_pos[1]+boxes_size[0]+2*spacing)

        self.right_components = [self.fps_label, self.fps_box, self.width_box, self.height_box, self.automaton_dropdown]

    def _generate_auto_controls_title(self, start_position):
        self.automaton_controls_title = TextLabel("Automaton controls :", fract_position=start_position, fract_width=0.2, font=self.font_title, color=(230, 89, 89), bg_color=(0,0,0,150), h_margin=0.2)
        self.automaton_controls_title.compute_size(self.sH, self.sW)
        self.extra_components_pos = (start_position[0], start_position[1]+self.automaton_controls_title.f_size[0]+0.007)

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
        
        # Live automaton label
        self.auto_label = TextLabel(text = self.auto.get_string_state(),
                                    fract_position=(0.02, 0.95), fract_width=0.8,color=(180,220,180),
                                    h_margin=0.2, bg_color=(0,0,0,150), font=self.font_text)
        
        self.left_components.append(self.auto_label)

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
            auto = AUTOMATAS[automaton_name]((self.H,self.W),**defaults, device=self.device)
            auto.set_components_fract_pos(self.extra_components_pos)
            
            self.automaton_controls_title.visible = len(auto._components)>0 and self.display_help.right_pane

            return auto
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
            self._base_events(event)
            self.auto._process_event_focus_check(event, self.camera)
            self.auto._process_gui_event(event)
            self._gui_events(event)
        
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
            

            self.auto.draw_components(self.screen)
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
            component._draw(self.screen)

        self.auto_label.text = self.auto.get_string_state() # Update live automaton label
        self.fps_label.text = f"FPS: {round(self.clock.get_fps())}"


        if(self.tablet_mode):
            for component in self.tablet_gui_components:
                component._draw(self.screen)

        if(self.tablet_mode):
            self.hide_show._draw(self.screen) # Always draw the hide/show button in tablet mode


        self.automaton_controls_title._draw(self.screen)

        for component in self.right_components:
            component._draw(self.screen)

    def _base_events(self,event):
        """
            Handles the base events, which are not automaton-specific.
        """
        if(not BaseComponent.get_focus_manager().should_process_event(event)):
            print('do NOT process base event')
            return
        
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
                    self.vid_writer = launch_video((self.H, self.W), self.video_fps, 'mp4v', vid_name=f'{self.auto.name()}')
            if(event.key == pygame.K_p):
                print_screen(self.auto.worldsurface, img_name=f'{self.auto.name()}')
            if(event.key == pygame.K_s):
                self.auto.step()
            if (event.key == pygame.K_h):
                self._hidden_pressed()
            if (event.key == pygame.K_c):
                self.camera.resize(self.sW,self.sH)
                self.camera.zoom = min(self.sW/self.W,self.sH/self.H) # Reset zoom to full view
                self.camera.center()
        
        if event.type == pygame.VIDEORESIZE:
            self.sW, self.sH = event.w, event.h
            self.s_size = (self.sW, self.sH)
            self.camera.resize(self.sW, self.sH)

    def _gui_events(self,event):
        """
            Handles the base GUI events, for fps, world size and automaton selection.
        """
        if self.automaton_dropdown._handle_event(event):
            selected = self.automaton_dropdown.selected
            self.auto = self.load_automaton(selected)
            self._generate_and_place_left_texts() # Need to update the text labels

        if self.fps_box._handle_event(event):
            try:
                self.fps = int(self.fps_box.value)
            except ValueError:
                print(f"Invalid FPS value: {self.fps_box.value}. Must be a positive integer.")
        
        if self.width_box._handle_event(event):
            # TODO : find a way for this to be automatic in the automaton (i.e., by default resize and reset)
            # And that way, we can override to do smart resizing
            if(self.width_box.value.strip() == ''):
                self.width_box.value = str(self.W)
                return
            self.W = int(self.width_box.value)
            print(f"Resizing automaton to width {self.W}")
            self.auto = self.load_automaton(self.automaton_dropdown.selected)  # Reload the automaton with the new width
            self.camera.change_border((self.W, self.H))  # Update the camera border size
        
        if self.height_box._handle_event(event):
            if(self.height_box.value.strip() == ''):
                self.height_box.value = str(self.W)
                return
            self.H = int(self.height_box.value)

            self.auto = self.load_automaton(self.automaton_dropdown.selected)  # Reload the automaton with the new height
            self.camera.change_border((self.W, self.H))  # Update the camera border size
        
        if(self.tablet_mode):
            if(self.play_pause._handle_event(event)):
                self.stopped = not self.stopped
            if(self.step._handle_event(event)):
                self.auto.step()
            if(self.hide_show.handle_event(event)):
                self._hidden_pressed()
                    

            # if(self.cam_reset.handle_event(event)):
            #     self.camera.resize(self.sW,self.sH)
            #     self.camera.zoom = min(self.sW/self.W,self.sH/self.H) # Reset zoom to full view
            #     self.camera.center()

    def _hidden_pressed(self):
        self.display_help.toggle()
        for component in self.right_components:
            component.visible = self.display_help.right_pane
            self.auto.set_components_visibility(self.display_help.right_pane)

            self.automaton_controls_title.visible = (self.display_help.right_pane and len(self.auto._components)>0)
        for component in self.left_components:
            component.visible = self.display_help.left_pane