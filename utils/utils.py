import os, torch, cv2, numpy as np
from showtens import show_image
import pygame

def launch_video(size,fps,fourcc='avc1'):
    """
        Returns Videowriter, ready to record and save the video.

        Parameters:
        size: (H,W) 2-uple size of the video
        fps: int, frames per second
        fourcc : Encoder, must work with .mp4 videos
    """
    os.makedirs('Videos',exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    numvids = len(os.listdir('Videos/'))
    vid_loc = f'Videos/Vid_{numvids}.mp4'
    return cv2.VideoWriter(vid_loc, fourcc, fps, (size[1], size[0]))

def add_frame(writer,worldmap):
    frame = worldmap.transpose(1,0,2) # (H,W,3)
    # tempB = np.copy(frame[:,:,2])
    # frame[:,:,2]=frame[:,:,0]
    # frame[:,:,0]=tempB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)

def save_image(worldmap):
    os.makedirs('Images',exist_ok=True)
    numimgs = len(os.listdir('Images/'))
    img_name = f'img_{numimgs}'
    show_image(torch.tensor(worldmap).permute(2,1,0),folderpath='Images',name = img_name)

def blit_text(surface, text, pos, font, color=pygame.Color('white')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    
    # Calculate total height of text
    total_height = 0
    for line in words:
        if line:  # Check if line is not empty
            word_height = font.render(line[0], 0, color).get_height()
            total_height += word_height

    # Handle special positions
    if pos == 'below_sx':
        x = 30  # Increased margin from left edge
        y = max_height - total_height - 30  # Increased margin from bottom
    elif pos == 'up_sx':
        x = 30  # Increased margin from left edge
        y = 30  # Increased margin from top
    else:
        x, y = pos

    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = 30 if pos in ['below_sx', 'up_sx'] else pos[0]  # Reset x with increased margin
                y += word_height  # Start on new row
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = 30 if pos in ['below_sx', 'up_sx'] else pos[0]  # Reset x with increased margin
        y += word_height  # Start on new row