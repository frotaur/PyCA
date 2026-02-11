from .utils import Camera, launch_video, add_frame, print_screen
# from .MainWindow import MainWindow
from .ui_components import *
import warnings
# remove super annoying warnings from resizing fonts
warnings.filterwarnings("ignore", category=UserWarning, module="pygame_gui")