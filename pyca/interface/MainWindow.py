"""
    Class for the main window of PyCA. Deals with the main GUI components,
    main pygame loop, and event handling and dispatching.

    Exposes an API of methods to modify stuff, such as giving the list of active automata,
    running the loop, and default values.
"""
import pygame
from importlib.resources import files

import pygame_gui

from pyca.interface import Camera, launch_video, print_screen, add_frame
from .utils.help_enum import HelpEnum
from ..automata import AUTOMATAS, Automaton
from .files import DEFAULTS, INTERFACE_HELP, BASE_FONT_PATH
from .ui_components import VertContainer, Button, TextLabel, InputField, DropDown, TextBox, BoxHolder, HorizContainer

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

        # self.font_text = SmartFont(fract_font_size=self.text_f_size, font_path=BASE_FONT_PATH)
        # self.font_title = SmartFont(fract_font_size=self.title_f_size, font_path=BASE_FONT_PATH)

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

        self._initial_automaton = "ElementaryCA"

        ## Right Panel Base Component
        # Define a position where we can put extra components. Its moved appropriately as we add stuff
        # self.right_components = None
        theme_path = files('pyca.interface.ui_components.styling').joinpath('theme.json')

        self.manager = pygame_gui.UIManager(window_resolution=(self.sW, self.sH), theme_path=str(theme_path))
        self.auto_gui = None

        # Load the initial automaton
        self._generate_right_base_gui()

        self.auto = self.load_automaton(self._initial_automaton)
        self._generate_left_base_gui()

        # if(self.tablet_mode):
        #     self.tablet_gui_components = None
        #     self._generate_tablet_gui(start_position=self.extra_components_pos)

        
    def _generate_tablet_gui(self, start_position=(0.8,0.1)):
        """
            Generates the tablet-mode GUI components, which are buttons for play/pause,
            step, reset, and a dropdown for automaton selection.
        """
        pass
        # H_SPACING = 0.007
        # W_SPACING = 0.01
        # BUTTONS_SIZE = (0.05,0.05)
        # next_pos = start_position
        # self.play_pause = Toggle(state1="Run", state2="Stop", fract_position=next_pos, fract_size=BUTTONS_SIZE)
        # next_pos = (next_pos[0]+BUTTONS_SIZE[1]+W_SPACING, next_pos[1])
        # self.step = Button(text="Step", fract_position=next_pos, fract_size=BUTTONS_SIZE)
        # next_pos = (next_pos[0]+BUTTONS_SIZE[1]+W_SPACING, next_pos[1])
        # self.hide_show = MultiToggle(states=["Hide Some", "Hide All", "Show"], fract_position=next_pos, fract_size=BUTTONS_SIZE,
        #                state_bg_colors=[(100, 100, 100), (20, 20, 80), (80, 20, 20)])
        # # For now, cam reset not needed as we can't move the camera in tablet mode
        # # self.cam_reset = Button(text="Center", fract_position=next_pos, fract_size=BUTTONS_SIZE)
        # next_pos = (start_position[0], next_pos[1]+BUTTONS_SIZE[0]+H_SPACING)
        # self.tablet_gui_components = [
        #     self.play_pause,
        #     self.step,
        #     self.hide_show
        # ]

        # self.extra_components_pos = (next_pos[0], next_pos[1])
    
    def _generate_right_base_gui(self):
        """
            Generates the base GUI which go on the right side of the window
        """
        self.right_box = BoxHolder(manager=self.manager, parent=None, rel_pos=(0.78,0.), rel_size=(1.0,0.22))
        self.right_container = VertContainer(manager=self.manager, parent=self.right_box, rel_pos=(0,0), rel_size=(1.0,1.0))
        ### FPS live label        
        self.live_fps_label = TextBox(f"FPS: {self.fps}", manager=self.manager, parent=self.right_container, rel_pos=(0,0), rel_size=(-1,1.), font_size=12, text_align='right', font_color=(230, 120, 120))

        ### FPS input box
        self.fps_box = HorizContainer(manager=self.manager, parent=self.right_container, rel_pos=(0,0), rel_size=(0.05,1.0),rel_padding=0.03)
        self.fps_input = InputField(manager=self.manager, parent=self.fps_box, rel_pos=(0,0), rel_size=(1.0,0.3), init_text=str(self.fps), allowed_chars=[str(i) for i in range(10)], max_length=3)
        self.width_input = InputField(manager=self.manager, parent=self.fps_box, rel_pos=(0,0), rel_size=(1.0,0.3), init_text=str(self.W), allowed_chars=[str(i) for i in range(10)], max_length=4)
        self.height_input = InputField(manager=self.manager, parent=self.fps_box, rel_pos=(0.,0), rel_size=(1.0,0.3), init_text=str(self.H), allowed_chars=[str(i) for i in range(10)], max_length=4)
        self.fps_box.add_component(self.fps_input)
        self.fps_box.add_component(self.width_input)
        self.fps_box.add_component(self.height_input)

        self.right_container.add_component(self.live_fps_label)
        self.right_container.add_component(self.fps_box)

        # Dropdown for automaton selection
        self.automaton_dropdown = DropDown(options=list(AUTOMATAS.keys()), manager=self.manager, parent=self.right_box, rel_pos=(0.0,0.94), rel_size=(0.06,1.0), open_upward=True)

    def _generate_left_base_gui(self):
        """
        Generates and places the left text labels of the main GUI. Needs to do some hacking to get dynamic positions
        of the texts, because a TextLabel component's height cannot be computed before it is rendered (because of text wrapping).
        """
        title_color = (230, 89, 89)
        description_color = (74, 101, 176)

        title_size = 16

        self.left_box = BoxHolder(manager=self.manager, parent=None, rel_pos=(0,0), rel_size=(1.0,0.22))
        self.left_text_container = VertContainer(manager=self.manager, parent=self.left_box, rel_pos=(0,0), rel_size=(1.0,1.0))

        self.automaton_name = TextBox(self.auto.name(),manager=self.manager, parent=self.left_text_container,rel_pos=(0,0), rel_size=(-1,1.),font_size=title_size, text_align='center',font_color=title_color)
        auto_desc, auto_help = self.auto.get_help()
        self.automaton_text = TextBox(auto_desc.strip(),manager=self.manager, parent=self.left_text_container,rel_pos=(0.,0.), rel_size=(-1,1.),font_color=description_color)
        controls_title = TextBox("General Controls",manager=self.manager, parent=self.left_text_container,rel_pos=(0.,0.01), rel_size=(-1,1.), font_size=title_size, text_align='center', font_color=title_color)
        controls = TextBox(INTERFACE_HELP['content'].strip(),manager=self.manager, parent=self.left_text_container,rel_pos=(0.,0.), rel_size=(-1,1.))
        automaton_help_title = TextBox("Automaton Controls",manager=self.manager, parent=self.left_text_container,rel_pos=(0.,0.), rel_size=(-1,1.),font_size=title_size, text_align='center', font_color=title_color)  
        self.automaton_help = TextBox(auto_help.strip(),manager=self.manager, parent=self.left_text_container,rel_pos=(0.,0.), rel_size=(-1,1.))

        self.left_text_container.add_component(self.automaton_name)
        self.left_text_container.add_component(self.automaton_text)
        self.left_text_container.add_component(controls_title)
        self.left_text_container.add_component(controls)
        self.left_text_container.add_component(automaton_help_title)
        self.left_text_container.add_component(self.automaton_help)

        self.live_auto_label = TextBox(text = self.auto.get_string_state(), manager=self.manager, parent=self.left_box,
                                       rel_pos=(0.,0.9),rel_size=(-1,1.), font_size=12, text_align='right')

    def _update_left_texts(self):
        """
            Updates the left text labels, for when the automaton changes.
        """
        self.automaton_name.text = self.auto.name()
        auto_desc, auto_help = self.auto.get_help()
        self.automaton_text.text = auto_desc.strip()
        self.automaton_help.text = auto_help.strip()

    def load_automaton(self, automaton_name):
        """
            Loads and returns the specified automaton model.
            Args:
                automaton_name (str): Name of the automaton model to load.
        """
        if(self.auto_gui is not None):
            self.right_container.remove_component(self.auto_gui)
        
        if automaton_name in AUTOMATAS:
            if(automaton_name in DEFAULTS): 
                defaults = DEFAULTS[automaton_name]
            else:
                defaults = {}
            auto : Automaton = AUTOMATAS[automaton_name]((self.H,self.W),**defaults, device=self.device)
            self.auto_gui = auto.get_gui_component()
            self.right_container.add_component(self.auto_gui)
            
            # self.automaton_controls_title.visible = len(auto._components)>0 and self.display_help.right_pane

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
            # self.auto._process_gui_event(event)
            self._gui_events(event)
            self.manager.process_events(event)
        
    def main_loop(self):
        """
            Runs the PyCA main loop.
        """
        while self.running:
            time_delta = self.clock.tick(self.fps)/1000.0
            self.handle_events()
            self.manager.update(time_delta)

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
            
            
            self.left_box._render()
            self.right_box._render()
            self.update_live_text()
            self.manager.draw_ui(self.screen)
            # self.auto.draw_components(self.screen)


            pygame.display.flip()
            
        if(self.vid_writer is not None):
            self.vid_writer.release()

        pygame.quit()
    
    def update_live_text(self):
        """
            Draws the help text on the screen.
        """
        self.live_auto_label.text = self.auto.get_string_state() # Update live automaton label
        self.live_fps_label.text = f"FPS: {round(self.clock.get_fps())}"


        # if(self.tablet_mode):
        #     for component in self.tablet_gui_components:
        #         component._draw(self.screen)

        # if(self.tablet_mode):
        #     self.hide_show._draw(self.screen) # Always draw the hide/show button in tablet mode

    def _base_events(self,event):
        """
            Handles the base events, which are not automaton-specific.
        """
        # if(not BaseComponent.get_focus_manager().should_process_event(event)):
        #     print('do NOT process base event')
        #     return
        
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
            self.manager.set_window_resolution((self.sW, self.sH))

    def _gui_events(self,event):
        """
            Handles the base GUI events, for fps, world size and automaton selection.
        """
        if self.automaton_dropdown.handle_event(event):
            selected = self.automaton_dropdown.selected
            self.auto = self.load_automaton(selected)
            self._update_left_texts()

        if self.fps_input.handle_event(event):
            try:
                self.fps = int(self.fps_input.value)
            except ValueError:
                print(f"Invalid FPS value: {self.fps_input.value}. Must be a positive integer.")
        
        if self.width_input.handle_event(event):
            # TODO : find a way for this to be automatic in the automaton (i.e., by default resize and reset)
            # And that way, we can override to do smart resizing
            if(self.width_input.value.strip() == ''):
                self.width_input.value = str(self.W)
                return
            self.W = int(self.width_input.value)
            print(f"Resizing automaton to width {self.W}")
            self.auto = self.load_automaton(self.automaton_dropdown.selected)  # Reload the automaton with the new width
            self.camera.change_border((self.W, self.H))  # Update the camera border size
        
        if self.height_input.handle_event(event):
            if(self.height_input.value.strip() == ''):
                self.height_input.value = str(self.H)
                return
            self.H = int(self.height_input.value)

            self.auto = self.load_automaton(self.automaton_dropdown.selected)  # Reload the automaton with the new height
            self.camera.change_border((self.W, self.H))  # Update the camera border size
        
        if(self.tablet_mode):
            if(self.play_pause.handle_event(event)):
                self.stopped = not self.stopped
            if(self.step.handle_event(event)):
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