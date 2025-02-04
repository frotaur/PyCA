"""
    Contains the main loop, used to run and visualize the automaton dynamics.
"""


import pygame, os
from utils.Camera import Camera
from Automata.models import CA1D, GeneralCA1D, CA2D, Baricelli1D, \
Baricelli2D, ReactionDiffusion, LGCA, FallingSand, NCA, MultiLenia
from Automata.models.ReactionDiffusion import GrayScott, BelousovZhabotinsky, Brusselator

from utils.utils import launch_video, add_frame, save_image
from interface import TextBlock, render_text_blocks, load_std_help



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
font = pygame.font.Font("public/fonts/AldotheApache.ttf", size=text_size)


screen = pygame.display.set_mode((sW,sH), flags=pygame.RESIZABLE)
clock = pygame.time.Clock() 
running = True
camera = Camera(W,H)
camera.resize(sW,sH)
zoom = min(sW,sH)/min(W,H)
camera.zoom = zoom

# Define here the automaton. Should be a subclass of Automaton, and implement 'draw()' and 'step()'.
# draw() should update the (3,H,W) tensor self._worldmap, for the visualization
#################   MULTICOLOR OUTER TOTALISTIC   ##################
r = 3
k = 3
random = True

# auto = GeneralCA1D((H,W),wolfram_num=1203,r=r,k=k,random=random) 
################################################################

#################   ELEMENTARY CA   #################################
# auto = CA1D((H,W),wolfram_num=90,random=True) 
################################################################


#################   BARICELLI   ####################################

#################   1D   ###########################################
# auto = Baricelli1D((H,W),n_species=8,reprod_collision=True)

#################   2D   ###########################################
# auto = Baricelli2D((H,W),n_species=7,reprod_collision=True,device=device)
################################################################

#################   CA2D   #################################
# auto = CA2D((H,W),b_num='3',s_num='23',random=True,device=device)
################################################################


################# Reaction Diffusion ############################
# auto = GrayScott((H,W),device=device)
# auto = BelousovZhabotinsky((H,W),device=device)
# auto = Brusselator((H,W),device=device)
################################################################

#################   LGCA   #################################
auto = LGCA((H,W), device=device)
################################################################

#################   Falling Sand   #################################
# auto = FallingSand((H,W))
################################################################

#################   NCA   #################################
# model_location = os.path.join('NCA_train','trained_model','latestNCA.pt')
# auto = NCA((H,W), model_path=model_location,device=device)
################################################################

################# Lenia ############################
# auto = MultiLenia((H,W),param_path='LeniaParams',device=device)

# Booleans for the main loop
stopped=True
recording=False
launch_vid=True
display_help=True
writer=None

description, help_text = auto.get_help()
std_help = load_std_help()
text_blocks = [
    TextBlock(description, "up_sx", (74, 101, 176), font),
    TextBlock("\n", "up_sx", (230, 230, 230), font)
]
for section in std_help['sections']:
    text_blocks.append(TextBlock(section["title"], "up_sx", (230, 89, 89), font))
    for command, description in section["commands"].items():
        text_blocks.append(TextBlock(f"{command} -> {description}", "up_sx", (230, 230, 230), font))
    text_blocks.append(TextBlock("\n", "up_sx", (230, 230, 230), font))
text_blocks.append(TextBlock("Automaton controls", "below_sx", (230, 89, 89), font))
text_blocks.append(TextBlock(help_text, "below_sx", (230, 230, 230), font))


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
                camera = Camera(W,H)
                camera.resize(sW,sH)
                zoom = min(sW,sH)/min(W,H)
                camera.zoom = zoom

        if event.type == pygame.VIDEORESIZE:
            camera.resize(event.w, event.h)

            
        auto.process_event(event,camera) # Process the event in the automaton

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
            writer = launch_video((H,W),fps,'H264')
        add_frame(writer,world_state) # (in the future, we may add the zoomed frame instead of the full frame)
        pygame.draw.circle(screen, (255,0,0), (15, H-15), 5)
    
    if (display_help):
        render_text_blocks(screen, [TextBlock(f"FPS: {int(clock.get_fps())}", "up_dx", (255, 89, 89), font)])
        render_text_blocks(screen, text_blocks)

    render_text_blocks(screen, [TextBlock(f"H for help", "below_dx", (89, 89, 89), font)])
    # Update the screen
    pygame.display.flip()

    clock.tick(fps)  # limits FPS to 60
    print('FPS : ', clock.get_fps(), end='\r')

pygame.quit()
