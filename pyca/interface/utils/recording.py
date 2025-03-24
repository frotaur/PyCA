import os, torch, cv2, numpy as np
from showtens import show_image,save_image
import pygame
import json

def launch_video(size,fps,fourcc='avc1'):
    """
        Returns Videowriter, ready to record and save the video.

        Parameters:
        size: (H,W) 2-uple size of the video
        fps: int, frames per second
        fourcc : Encoder, must work with .mp4 videos
    """
    os.makedirs('videos',exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    numvids = len(os.listdir('videos/'))
    vid_loc = f'videos/vid_{numvids}.webm'
    return cv2.VideoWriter(vid_loc, fourcc, fps, (size[1], size[0]))

def add_frame(writer, worldmap):
    frame = worldmap.transpose(1,0,2) # (H,W,3)
    # tempB = np.copy(frame[:,:,2])
    # frame[:,:,2]=frame[:,:,0]
    # frame[:,:,0]=tempB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)

def print_screen(worldmap):
    os.makedirs('images',exist_ok=True)
    numimgs = len(os.listdir('images/'))
    img_name = f'img_{numimgs}'
    save_image(torch.tensor(worldmap).permute(2,1,0),folder='images',name = img_name)

