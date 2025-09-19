import os, torch, cv2
from showtens import save_image
import pygame
from pathlib import Path

def launch_video(size,fps,fourcc='avc1', vid_name='vid', output_folder='./data/videos/'):
    """
        Returns Videowriter, ready to record and save the video.

        Parameters:
        size: (H,W) 2-uple size of the video
        fps: int, frames per second
        fourcc : Encoder, must work with .mp4 videos
    """
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    vid_loc = free_video_name(output_folder, vid_name)

    return cv2.VideoWriter(str(vid_loc), fourcc, fps, (size[1], size[0]))

def add_frame(writer, worldsurface):
    worldmap = pygame.surfarray.array3d(worldsurface) # (W,H,3)

    frame = worldmap.transpose(1,0,2) # (H,W,3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)

def print_screen(worldsurface, output_folder='./data/images/', img_name='img'):
    worldmap = pygame.surfarray.array3d(worldsurface) # (W,H,3)
    
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    
    free_path = free_video_name(output_folder, img_name)

    save_image(torch.tensor(worldmap,dtype=float).permute(2,1,0)/255., folder=free_path.parent, name=free_path.name.split('.')[0])

def free_video_name(output_folder, vid_name):
    """
        Returns a free video name in the specified folder, with the specified base name.
        
        Args:
        output_folder : str
            The folder where to look for existing videos.
        vid_name : str
            The base name of the video.
        
        Returns:
        Path object containing the full path to next available video name.
    """
    
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    # Find next available number
    counter = 0
    while True:
        vid_loc = path / f'{vid_name}_{counter}.mp4'
        if not vid_loc.exists():
            break
        counter += 1

    return vid_loc