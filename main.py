"""
    Contains the main loop, used to run and visualize the automaton dynamics.
"""


import pygame
from Camera import Camera
from CA1D import CA1D, GeneralCA1D
from Baricelli import Baricelli1D, Baricelli2D
from utils import launch_video, add_frame, save_image
pygame.init()
W,H =800,600 # Width and height of the window
fps = 60 # Visualization (target) frames per second

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

auto = GeneralCA1D((H,W),wolfram_num=1203,r=r,k=k,random=random) 
################################################################

#################   ELEMENTARY CA   #################################
# auto = CA1D((H,W),wolfram_num=90,random=True) 
################################################################

#################   BARICELLI   ####################################

#################   1D   ###########################################
# auto = Baricelli1D((H,W),n_species=6,reprod_collision=False)

#################   2D   ###########################################
auto = Baricelli2D((H,W),n_species=7,reprod_collision=True,device='cuda')
################################################################
# Booleans for mouse events
stopped=True
add_drag = False
rem_drag = False
recording=False
launch_vid=True

writer=None

alt= 0
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
    
        auto.process_event(event,camera) # Process the event in the automaton


    if(not stopped):
        auto.step() # step the automaton
    
    auto.draw() # draw the worldstate
        
    world_state = auto.worldmap
    surface = pygame.surfarray.make_surface(world_state)

    if(recording):
        if(launch_vid):# If the video is not launched, we create it
            launch_vid = False
            writer = launch_video((H,W),fps,'H265')
        add_frame(writer,world_state) # (in the future, we may add the zoomed frame instead of the full frame)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))

    # blit a red circle down to the left when recording
    if(recording):
        pygame.draw.circle(screen, (255,0,0), (15, H-15), 5)
    
    # Update the screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60


pygame.quit()
