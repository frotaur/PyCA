import pygame, os, torch
from utils.Camera import Camera
from Automata.models import (
    CA1D, 
    GeneralCA1D, 
    CA2D, 
    Baricelli1D,
    Baricelli2D, 
    LGCA, 
    FallingSand, 
    MultiLenia,
    NCA
)
from Automata.models.ReactionDiffusion import (
    GrayScott, 
    BelousovZhabotinsky, 
    Brusselator
)

from utils.utils import launch_video, add_frame, save_image
from interface.text import TextBlock, DropdownMenu, InputField, render_text_blocks, load_std_help


if os.name == 'posix':  # Check if OS is Linux/Unix
    print("Setting window position to 0, 0")
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0, 0"

pygame.init()

# Screen size
sW, sH = 1280, 720

# Automaton world size 
W, H = 300, 300

# Device to run the automaton
device = 'cuda'

fps = 400 # Visualization (target) frames per second
text_size = int(sH/40)
title_size = int(text_size*1.5)
font = pygame.font.Font("public/fonts/AldotheApache.ttf", size=text_size)
font_title = pygame.font.Font("public/fonts/AldotheApache.ttf", size=title_size)
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

# Define automaton classes without instantiating
automaton_options = {
    "CA2D":         lambda h, w: CA2D((h,w), b_num='3', s_num='23', random=True, device='cuda'),
    "CA1D":         lambda h, w: CA1D((h,w), wolfram_num=90, random=True),
    "GeneralCA1D":  lambda h, w: GeneralCA1D((h,w), wolfram_num=1203, r=3, k=3, random=True),
    "LGCA":         lambda h, w: LGCA((h,w), device='cuda'),
    "Gray-Scott":   lambda h, w: GrayScott((h,w), device='cuda'),
    "Belousov-Zhabotinsky": lambda h, w: BelousovZhabotinsky((h,w), device='cuda'),
    "Brusselator":  lambda h, w: Brusselator((h,w), device='cuda'),
    "Falling Sand": lambda h, w: FallingSand((h,w)),
    "Baricelli 2D": lambda h, w: Baricelli2D((h,w), n_species=7, reprod_collision=True, device='cuda'),
    "Baricelli 1D": lambda h, w: Baricelli1D((h,w), n_species=8, reprod_collision=True),
    "MultiLenia":   lambda h, w: MultiLenia((h,w), param_path='LeniaParams', device='cuda'),
    # "Neural CA":  lambda h, w: NCA((h,w), model_path='NCA_train/trained_model/latestNCA.pt', device='cuda')
}

# Then when initializing the first automaton:
initial_automaton = "CA2D"
auto = automaton_options[initial_automaton](H, W)

description, help_text = auto.get_help()
std_help = load_std_help()

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

dropdown = DropdownMenu(
    screen=screen,
    width=200,
    height=30,
    font=font,
    options=automaton_options,
    default_option="CA2D",
    margin=20  # Distance from screen edges
)

# Create input fields for width and height
w_input = InputField(
    screen=screen,
    width=60,
    height=30,
    font=font,
    label="Width",
    initial_value=W,
    margin=20,
    index=0  # First input field
)

h_input = InputField(
    screen=screen,
    width=60,
    height=30,
    font=font,
    label="Height",
    initial_value=H,
    margin=20,
    index=1  # Second input field, will appear below width
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
                save_image(auto.worldmap)
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
            camera.resize(event.w, event.h)
            dropdown.update_position()
            w_input.update_position()
            h_input.update_position()
            text_size = int(event.h/45)
            font = pygame.font.Font("public/fonts/AldotheApache.ttf", size=text_size)
            font_title = pygame.font.Font("public/fonts/AldotheApache.ttf", size=title_size)
            text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)
        
        auto.process_event(event,camera) # Process the event in the automaton

        if dropdown.handle_event(event):
            # Handle automaton change
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            auto = automaton_options[dropdown.current_option](H, W)
            # Update help text
            description, help_text = auto.get_help()
            text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)

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

    if(not stopped):
        auto.step() # step the automaton
    
    auto.draw() # draw the worldstate
    world_state = auto.worldmap
    surface = pygame.surfarray.make_surface(world_state)
    
    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)
    screen.blit(zoomed_surface, (0,0))

    if (recording):
        if(launch_vid):# If the video is not launched, we create it
            launch_vid = False
            writer = launch_video((H,W), fps, 'mp4v')
        add_frame(writer,world_state) # (in the future, we may add the zoomed frame instead of the full frame)
        pygame.draw.circle(screen, (255,0,0), (15,H-15), 5)
    
    if (display_help):
        render_text_blocks(screen, [TextBlock(f"FPS: {int(clock.get_fps())}", "up_dx", (255, 89, 89), font)])
        render_text_blocks(screen, text_blocks)

    render_text_blocks(screen, [TextBlock(f"H -> help", "below_dx", (74, 101, 176), font)])

    # Draw dropdown (before pygame.display.flip())
    dropdown.draw(screen)

    # Draw input fields
    w_input.draw()
    h_input.draw()

    # Update the screen
    pygame.display.flip()

    clock.tick(fps)  # limits FPS to 60
    print('FPS : ', clock.get_fps(), end='\r')

pygame.quit()
