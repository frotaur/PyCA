from .BaseComponent import BaseComponent
from .SmartFont import SmartFont
import pygame


class TextLabel(BaseComponent):
    """
    Represents a text label in the UI.
    """

    def __init__(self, text, fract_position, fract_width, font: SmartFont, color=(255, 89, 89)):
        """
        Initializes the text label component.

        Args:
            text (str): The text to display.
            screen_size (tuple): Size of the screen (height, width).
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_width (float): Fractional width in [0,1] of the component. Height is 
            determined by the font size
            font (SmartFont): Font object for rendering the text.
            color (tuple): Color of the text in RGB format.
        """
        fract_size = (0. , fract_width) # Height cannot be computed before the font renders
        super().__init__(fract_position, fract_size)
        self.text = text
        self.font = font
        self.color = color
        self.rendered_lines = []
    
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
                
            text_surface = self.font.font.render(test_text, True, self.color)
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
            text_surface = self.font.font.render(line, True, self.color)
            self.rendered_lines.append(text_surface)
    
    def draw(self, screen: pygame.Surface) -> pygame.Surface:
        """
        Efficiently draws pre-rendered text lines to the screen.
        Lines that exceed the component height are clipped.
        """
        super().draw(screen)
        
        # Simply blit the pre-rendered lines
        y_offset = 0
        line_height = self.font.font.get_height()

        for text_surface in self.rendered_lines:
            screen.blit(text_surface, (self.x, self.y + y_offset))
            y_offset += line_height

        return screen
        

    def handle_event(self, event:pygame.event.Event) -> bool:
        """
        Text labels do not handle events by default.
        Args:
            event (pygame.event.Event): The event to handle (optional)
        """
        return False