import pygame, os, cv2
from torch.cuda import empty_cache, reset_max_memory_allocated

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

from utils.Camera import Camera
from utils.utils import launch_video, add_frame, save_image
from interface.text import TextBlock, DropdownMenu, InputField, render_text_blocks, load_std_help

import imutils
from imutils.video import VideoStream
from flask import Response, Flask, render_template, request

if os.name == 'posix':  # Check if OS is Linux/Unix
    print("Setting window position to 0, 0")
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0, 0"

app = Flask(__name__)

viewport_size = {'width': 1920, 'height': 1080}  # Default size

@app.route("/")
def index():    
	return render_template("index.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(gameloop(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/set_viewport_size")
def set_viewport_size():
    global viewport_size
    # Add minimum size constraints
    width = max(int(request.args.get('width', 1920)), 640)  # Minimum width of 640px
    height = max(int(request.args.get('height', 1080)), 480)  # Minimum height of 480px
    viewport_size['width'] = width
    viewport_size['height'] = height
    return "OK"

@app.route("/keypress")
def keypress():
    key = request.args.get('key')
    key_mapping = {
        'space': pygame.K_SPACE,
        'r': pygame.K_r,
        's': pygame.K_s,
        'h': pygame.K_h,
        'c': pygame.K_c
    }
    
    if key in key_mapping:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': key_mapping[key]}))
    return "OK"

@app.route("/mouse_event")
def mouse_event():
    event_type = request.args.get('type')
    # Get coordinates relative to the video feed
    x = float(request.args.get('x', 0))
    y = float(request.args.get('y', 0))
    
    # Map browser coordinates to pygame coordinates
    # We need to account for the aspect ratio and scaling
    pygame_x = int((x / viewport_size['width']) * viewport_size['width'])
    pygame_y = int((y / viewport_size['height']) * viewport_size['height'])
    
    # Create and post the appropriate pygame event
    if event_type == 'wheel':
        delta = int(request.args.get('delta', 0))
        # Wheel up is negative delta, down is positive
        button = 5 if delta > 0 else 4
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {
                'button': button,
                'pos': (pygame_x, pygame_y)
            }
        ))
    elif event_type == 'mousedown':
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {
                'button': 1,
                'pos': (pygame_x, pygame_y)
            }
        ))
    elif event_type == 'mousemove':
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEMOTION,
            {
                'pos': (pygame_x, pygame_y),
                'rel': (0, 0),  # We don't track relative movement for now
                'buttons': (1, 0, 0)  # Left button pressed
            }
        ))
    
    return "OK"

pygame.init()

def gameloop():
    global viewport_size
    # Replace the static sW, sH definition with:
    sW, sH = viewport_size['width'], viewport_size['height']

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
                empty_cache()
                reset_max_memory_allocated()
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

        screen_surface = pygame.display.get_surface()
        screen_array = pygame.surfarray.array3d(screen_surface)
        screen_array = screen_array.transpose([1,0,2])
        screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)

        # Before encoding, optionally resize if needed for performance
        # screen_array = cv2.resize(screen_array, (1280, 720))  # Uncomment if you want to limit resolution

        # Improve JPEG encoding quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]  # Quality from 0-100, higher is better
        (flag, encoded_image) = cv2.imencode('.jpg', screen_array, encode_params)
        if not flag: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

        clock.tick(fps)  # limits FPS to 60
        print('FPS : ', clock.get_fps(), end='\r')

    pygame.quit()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)