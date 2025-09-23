from .BaseComponent import BaseComponent
from .SmartFont import SmartFont
import pygame


class TextLabel(BaseComponent):
    """
    Represents a text label in the UI.
    """

    def __init__(self, text, font: SmartFont, fract_position=(0,0), fract_width=0.1, line_spacing=0., color=(255, 89, 89), bg_color=(0,0,0,0),
                 h_margin=0.):
        """
        Initializes the text label component.

        Args:
            text (str): The text to display.
            screen_size (tuple): Size of the screen (height, width).
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_width (float): Fractional width in [0,1] of the component. Height is 
            determined by the font size
            font (SmartFont): Font object for rendering the text.
            line_spacing (float): Additional spacing between lines when text overflows. As fraction of font size.
            color (tuple): Color of the text in RGB format.
            bg_color (tuple, optional): Background color of the label in RGB or RGBA format. Defaults to None.
            h_margin (float): Height margin around the text in fractional units of the font size.
        """
        fract_size = (0. , fract_width) # Height cannot be computed before the font renders
        super().__init__(fract_position, fract_size)
        self._text = text
        self.font = font
        self.color = color
        self.bg_color = bg_color
        self.margin = h_margin
        self.line_spacing = line_spacing
        self.rendered_lines = []
    
    @property
    def text(self):
        """
        Returns the text of the label.
        """
        return self._text
    
    @text.setter
    def text(self, new_text):
        """
        Sets the text of the label and re-renders it (if possible). 
        """
        self._text = new_text
        if(self.sH is not None):
            self.render()
    
    def render(self):
        """
        Renders the text with correct font size and height/width.
        """
        self.font.sH = self.sH 
        self._prepare_text()

    def _prepare_text(self):
        """
        Preprocesses the text into wrapped lines and renders them as surfaces.
        Called once during initialization and when font size changes.
        """
            
        # Split text into words for wrapping
        words = self.text.replace('\n',' \n ').split(' ')
            
        lines = []
        current_line = []
        
        # Build lines that fit within the component width
        for word in words:
            if(word!= '\n'):
                test_line = current_line + [word]
                test_text = ' '.join(test_line)
            else:
                lines.append(' '.join(current_line))
                current_line = []
                continue
                
            text_surface = self.font.render(test_text, self.color)
            if text_surface.get_width() <= self.w:
                current_line = test_line
            else:
                # Current line is full, start a new one
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too wide, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Pre-render all lines as surfaces
        self.rendered_lines = []
        for line in lines:
            text_surface = self.font.render(line, self.color)

            self.rendered_lines.append(text_surface)

        self.f_size = (self.margin*2*self.font.f_height+len(self.rendered_lines) * (self.font.f_height*(1+self.line_spacing)), self.f_size[1]) # Can now compute the fractional height!

    def draw(self, screen: pygame.Surface) -> pygame.Surface:
        """
        Efficiently draws pre-rendered text lines to the screen.
        Lines that exceed the component height are clipped.
        """
        # Simply blit the pre-rendered lines
        y_offset = self.margin*self.font.height
        line_height = self.font.height*(1+self.line_spacing)

        max_width = max([line.get_width() for line in self.rendered_lines], default=0)
        full_surface = pygame.Surface((max_width, self.h), pygame.SRCALPHA)
        full_surface.fill(self.bg_color)
        for text_surface in self.rendered_lines:
            full_surface.blit(text_surface, (0, y_offset))
            y_offset += line_height

        screen.blit(full_surface, (self.x, self.y))
        return screen
        

    def handle_event(self, event:pygame.event.Event) -> bool:
        """
        Text labels do not handle events by default.
        Args:
            event (pygame.event.Event): The event to handle (optional)
        """
        return False

    def compute_size(self, sH,sW):
        """
        Given the screen size, renders the text to generate self.size and self.f_size
        
        Args:
            screen_size (tuple): Size of the screen (height, width).
        """
        self.set_screen_size(sH,sW)
        self.render()