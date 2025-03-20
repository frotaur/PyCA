import pygame, os, json
from torch.cuda import empty_cache, reset_max_memory_allocated
from importlib.resources import files

from pyca.interface import Camera
from pyca.automata.models import *
from pyca.interface import launch_video, add_frame, print_screen
from pyca.interface.text import TextBlock, DropdownMenu, InputField, render_text_blocks

if os.name == 'posix':  # Check if OS is Linux/Unix
    print("Setting window position to 0, 0")
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0, 0"

font_path = str(files('pyca.interface.files').joinpath('AldotheApache.ttf'))
std_help = None

with open(str(files('pyca.interface.files').joinpath('std_help.json')), 'r') as f:
    std_help = json.load(f)

pygame.init()

def gameloop(screen: tuple[int], world: tuple[int], device: str):
    # Define available automaton classes
    automaton_options = {
        "CA2D":         lambda h, w: CA2D((h,w), b_num='3', s_num='23', random=True, device=device),
        "ElementaryCA": lambda h, w: ElementaryCA((h,w), wolfram_num=90, random=True),
        "Rule110Universality": lambda h, w: Rule110Universality((h,w), wolfram_num=110, random=True),
        "Totalistic1DCA":lambda h, w: TotalisticCA1D((h,w), wolfram_num=1203, r=3, k=3, random=True),
        "LGCA":         lambda h, w: LGCA((h,w), device=device),
        "Gray-Scott":   lambda h, w: GrayScott((h,w), device=device),
        "Belousov-Zhabotinsky": lambda h, w: BelousovZhabotinsky((h,w), device=device),
        "Brusselator":  lambda h, w: Brusselator((h,w), device=device),
        "Falling Sand": lambda h, w: FallingSand((h,w)),
        "Baricelli 2D": lambda h, w: Baricelli2D((h,w), n_species=7, reprod_collision=True, device=device),
        "Baricelli 1D": lambda h, w: Baricelli1D((h,w), n_species=8, reprod_collision=True),
        "MultiLenia":   lambda h, w: MultiLenia((h,w), dt=0.1, num_channels=3, param_path='lenia_cool_params', device=device),
        "Neural CA":    lambda h, w: NCA((h,w), models_folder='saved_models/NCA/', device=device),
        "Von Neumann":  lambda h, w: VonNeumann((h,w),element_size=9, device=device),
    }

    
    # Replace the static sW, sH definition with:
    sW, sH = screen

    # Automaton world size 
    W, H = world

    # Device to run the automaton
    device = device

    fps = 60 # Visualization (target) frames per second
    video_fps = 60 # Video frames per second

    text_size = int(sH/45)
    title_size = int(text_size*1.3)
    font = pygame.font.Font(font_path, size=text_size)
    font_title = pygame.font.Font(font_path, size=title_size)

    programIcon = pygame.image.load(files('pyca.interface.files').joinpath('icon.png'))
    pygame.display.set_icon(programIcon)
    pygame.display.set_caption('PyCA')
    screen = pygame.display.set_mode((sW,sH), flags=pygame.RESIZABLE)
    clock = pygame.time.Clock() 
    running = True
    camera = Camera(W,H)
    camera.resize(sW,sH)
    zoom = min(sW/W,sH/H)
    camera.zoom = zoom

    # Booleans for the main loop
    stopped=True
    recording=False
    launch_vid=True
    display_help=True
    writer=None

    # Then when initializing the first automaton:
    initial_automaton = "CA2D"
    auto = automaton_options[initial_automaton](H, W)

    description, help_text = auto.get_help()

    def make_text_blocks(description, help_text, std_help, font, font_title):
        text_blocks = [
            TextBlock(description, "up_sx", (74, 101, 176), font_title),
            TextBlock("\n", "up_sx", (230, 230, 230), font)
        ]
        for section in std_help['sections']:
            text_blocks.append(TextBlock(section["title"], "up_sx", (230, 89, 89), font))
            for command, description in section["commands"].items():
                text_blocks.append(TextBlock(f"{command} -> {description}", "up_sx", (230, 230, 230), font))
            text_blocks.append(TextBlock("\n", "up_sx", (230, 230, 230), font))
        text_blocks.append(TextBlock("Automaton controls", "below_sx", (230, 89, 89), font))
        text_blocks.append(TextBlock(help_text, "below_sx", (230, 230, 230), font))
        return text_blocks

    text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)

    # Update these initial sizes to be relative to screen size
    button_width = int(sW * 0.15)  # 15% of screen width
    button_height = int(sH * 0.05)  # 5% of screen height
    input_width = int(sW * 0.05)   # 5% of screen width
    input_height = int(sH * 0.05)  # 5% of screen height
    margin = int(sH * 0.02)        # 2% of screen height


    dropdown = DropdownMenu(
        screen=screen,
        width=button_width,
        height=button_height,
        font=font,
        options=automaton_options,
        default_option=initial_automaton,
        margin=margin
    )

    # Update input fields with new relative sizes
    w_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="Width",
        initial_value=W,
        margin=margin,
        index=0
    )

    h_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="Height",
        initial_value=H,
        margin=margin,
        index=1
    )

    fps_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="FPS",
        initial_value=fps,
        margin=margin,
        index=2
    )


    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            camera.handle_event(event) # Handle the camera events

            if event.type == pygame.KEYDOWN :
                if(event.key == pygame.K_SPACE): # Press 'SPACE' to start/stop the automaton
                    stopped=not(stopped)
                if(event.key == pygame.K_q):
                    running=False
                if(event.key == pygame.K_r): # Press 'R' to start/stop recording
                    recording = not recording
                    if(not launch_vid and writer is not None):
                        launch_vid=True
                        writer.release()
                if(event.key == pygame.K_p):
                    print_screen(auto.worldsurface)
                if(event.key == pygame.K_s):
                    auto.step()
                if (event.key == pygame.K_h):
                    display_help = not display_help
                if (event.key == pygame.K_c):
                    current_sW, current_sH = screen.get_size()
                    camera = Camera(W,H)
                    camera.resize(current_sW,current_sH)
                    zoom = min(current_sW/W,current_sH/H)
                    camera.zoom = zoom

            if event.type == pygame.VIDEORESIZE:
                # Get current window size and new window size
                old_w, old_h = screen.get_size()
                new_w, new_h = event.w, event.h
                
                # Calculate scale factors
                scale_w = new_w / old_w
                scale_h = new_h / old_h
                
                # Update camera with new screen dimensions and scale position and zoom
                camera.resize(new_w, new_h)
                camera.position.x *= scale_w
                camera.position.y *= scale_h
                camera.zoom *= min(scale_w, scale_h)  # Use minimum scale to preserve aspect ratio
                camera.updateFov()
                
                # Calculate new sizes based on new dimensions
                button_width = int(new_w * 0.15)
                button_height = int(new_h * 0.05)
                input_width = int(new_w * 0.05)
                input_height = int(new_h * 0.05)
                margin = int(new_h * 0.02)
                
                # Update text sizes
                text_size = int(new_h/45)
                title_size = int(text_size*1.5)
                font = pygame.font.Font(font_path, size=text_size)
                font_title = pygame.font.Font(font_path, size=title_size)
                
                # Update UI elements with new sizes and font
                dropdown.resize(button_width, button_height, margin, font)
                w_input.resize(input_width, input_height, margin, font)
                h_input.resize(input_width, input_height, margin, font)
                fps_input.resize(input_width, input_height, margin, font)
                
                # Update text blocks with new font
                text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)
            
            auto.process_event(event,camera) # Process the event in the automaton

            if dropdown.handle_event(event):
                # # Handle automaton change
                auto = automaton_options[dropdown.current_option](H, W)
                # Update help text
                description, help_text = auto.get_help()
                text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)

            if(display_help):
                # Handle input field events
                if w_input.handle_event(event):
                    new_w = w_input.get_value()
                    if new_w and new_w > 0:
                        W = new_w
                        current_sW, current_sH = screen.get_size()
                        # Recreate automaton with new size
                        auto = automaton_options[dropdown.current_option](H, W)
                        camera = Camera(W,H)
                        camera.resize(current_sW,current_sH)
                        zoom = min(current_sW/W,current_sH/H)
                        camera.zoom = zoom

                if h_input.handle_event(event):
                    new_h = h_input.get_value()
                    if new_h and new_h > 0:
                        H = new_h
                        current_sW, current_sH = screen.get_size()
                        # Recreate automaton with new size
                        auto = automaton_options[dropdown.current_option](H, W)
                        camera = Camera(W,H)
                        camera.resize(current_sW,current_sH)
                        zoom = min(current_sW/W,current_sH/H)
                        camera.zoom = zoom

                if fps_input.handle_event(event):
                    new_fps = fps_input.get_value()
                    if new_fps and new_fps > 0:
                        fps = new_fps                    
                        


        if(not stopped):
            auto.step() # step the automaton
        
        auto.draw() # draw the worldstate
        world_surface = auto.worldsurface
        
        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the scaled surface on the window
        zoomed_surface = camera.apply(world_surface, border=True)
        screen.blit(zoomed_surface, (0,0))

        if (recording):
            if(launch_vid):# If the video is not launched, we create it
                launch_vid = False
                writer = launch_video((H,W), video_fps, 'mp4v')
            add_frame(writer,world_surface) # (in the future, we may add the zoomed frame instead of the full frame)
            pygame.draw.circle(screen, (255,0,0), (sW-10,15), 7)
        
        if (display_help):
            render_text_blocks(screen, [TextBlock(f"FPS: {int(clock.get_fps())}", "up_dx", (255, 89, 89), font)])
            render_text_blocks(screen, text_blocks)

        # Draw dropdown (before pygame.display.flip())
        dropdown.draw(screen, display_text=display_help)

        # Draw input fields
        if(display_help):
            w_input.draw()
            h_input.draw()
            fps_input.draw()
        # Update the screen
        pygame.display.flip()

        clock.tick(fps)  # limits FPS to 60

    pygame.quit()

if __name__=="__main__":
    gameloop((800,600), (100,100), 'cuda')