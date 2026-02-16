from importlib.resources import files
from .default_configs import DEFAULTS
from .ui_stylings import *

BASE_FONT_PATH = str(files(f'{__package__}') / 'AldotheApache.ttf')
