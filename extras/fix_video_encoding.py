"""
Script to re-encode the videos recorded so that they work everywhere.
Run it in the directory where the videos are located.
"""

import os
import subprocess
from pathlib import Path

def fix_video_encoding(input_path, speedup_factor=1.0):
    """
    Fix video encoding for web compatibility and optionally speed up the video.
    
    Args:
        input_path (str): Path to the input video file
        speedup_factor (float): Factor by which to speed up the video. 
                              1.0 = original speed, 2.0 = twice as fast, etc.
    
    Returns:
        str: Path to the output video file
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        subprocess.CalledProcessError: If FFmpeg command fails
    """
    # Convert input path to Path object for easier manipulation
    input_path = Path(input_path)
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output path with '_fixed' suffix
    output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    
    # Build FFmpeg command
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-vcodec', 'h264',
        '-acodec', 'aac'
    ]
    
    # Add speed modification if speedup_factor is not 1.0
    if speedup_factor != 1.0:
        # Use setpts filter to adjust video speed
        # 1/speedup_factor adjusts the presentation timestamp
        command.extend([
            '-filter:v', f'setpts={1/speedup_factor}*PTS',
            # Adjust audio speed without changing pitch
            '-filter:a', f'atempo={speedup_factor}'
        ])
    
    # Add output path to command
    command.append(str(output_path))
    
    try:
        # Run FFmpeg command
        subprocess.run(command, check=True, capture_output=True, text=True)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e.stderr}")
        raise

# Example usage:
if __name__ == "__main__":
    try:
        # Fix video encoding without speed change
        output_path = fix_video_encoding("vid_9.mp4")
        # output_path = fix_video_encoding("vid_30.mp4")
        print(f"Fixed video saved to: {output_path}")
        
        # # Fix video encoding and make it twice as fast
        # output_path = fix_video_encoding("input.mp4", speedup_factor=2.0)
        # print(f"Fixed and sped up video saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")