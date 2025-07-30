from importlib.resources import files
from .default_configs import DEFAULTS
from .help_strings import INTERFACE_HELP

BASE_FONT_PATH = str(files(f'{__package__}') / 'AldotheApache.ttf')
