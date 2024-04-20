"""
    Contains the main loop, used to run and visualize the automaton dynamics.
"""


import pygame
from Camera import Camera
from Automata.models import CA1D, GeneralCA1D, CA2D, Baricelli1D, Baricelli2D, ReactionDiffusion, LGCA, FallingSand, NCA
from Automata.models.ReactionDiffusion import GrayScott, BelousovZhabotinsky, Brusselator

from utils import launch_video, add_frame, save_image
pygame.init()
W,H = 300, 300 # Width and height of the window
fps = 120 # Visualization (target) frames per second

screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock() 
running = True
camera = Camera(W,H)

random = True
# Define here the automaton. Should be a subclass of Automaton, and implement 'draw()' and 'step()'.
# draw() should update the (3,H,W) tensor self._worldmap, for the visualization
#################   MULTICOLOR OUTER TOTALISTIC   ##################
r = 3
k = 3

# auto = GeneralCA1D((H,W),wolfram_num=1203,r=r,k=k,random=random) 
################################################################

#################   ELEMENTARY CA   #################################
# auto = CA1D((H,W),wolfram_num=90,random=True) 
################################################################


#################   BARICELLI   ####################################

#################   1D   ###########################################
# auto = Baricelli1D((H,W),n_species=8,reprod_collision=True)

#################   2D   ###########################################
# auto = Baricelli2D((H,W),n_species=7,reprod_collision=True,device='cuda')
################################################################

#################   CA2D   #################################
auto = CA2D((H,W),b_num='3',s_num='23',random=random,device='cuda')
################################################################

################# Reaction Diffusion ############################
# auto = ReactionDiffusion((H,W),device='cpu')
################################################################

################# Reaction Diffusion ############################
# auto = GrayScott((H,W),device='cuda')
# auto = BelousovZhabotinsky((H,W),device='cuda')
# auto = Brusselator((H,W),device='cuda')
################################################################

#################   LGCA   #################################
# auto = LGCA((H,W), device='cuda')
################################################################

#################   Falling Sand   #################################
# auto = FallingSand((H,W))
################################################################

#################   NCA   #################################
# auto = NCA((H,W), model_path='NCA_train/runs/NCANew/state/latestNCA.pt',device='cuda')
################################################################
# Booleans for the main loop
stopped=True
recording=False
launch_vid=True
writer=None

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
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
    
        auto.process_event(event,camera) # Process the event in the automaton


    if(not stopped):
        auto.step() # step the automaton
    
    auto.draw() # draw the worldstate
        
    world_state = auto.worldmap

    surface = pygame.surfarray.make_surface(world_state)

    if(recording):
        if(launch_vid):# If the video is not launched, we create it
            launch_vid = False
            writer = launch_video((H,W),fps,'H264')
        add_frame(writer,world_state) # (in the future, we may add the zoomed frame instead of the full frame)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))

    # blit a red circle down to the left when recording
    if(recording):
        pygame.draw.circle(screen, (255,0,0), (15, H-15), 5)

    # if hasattr(auto, 'brush_size'):
    #     pygame.draw.circle(screen, (128,50,50,50), camera.convert_mouse_pos(pygame.mouse.get_pos()), auto.brush_size)
    # Update the screen
    pygame.display.flip()

    clock.tick(fps)  # limits FPS to 60
    print('FPS : ', clock.get_fps(), end='\r')

pygame.quit()
