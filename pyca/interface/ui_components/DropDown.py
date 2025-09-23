from .BaseComponent import BaseComponent
from ..files import BASE_FONT_PATH
from .SmartFont import SmartFont
import pygame


class DropDown(BaseComponent):
    """
    Component for creating a dropdown/dropup menu, with text options to select from.
    Its state (self.selected) is simply a string. 
    """

    def __init__(self, options, fract_position=(0.,0.), fract_size=(0.05,0.1), open_upward=False, 
                    bg_color=(50, 50, 50), text_color=(230, 230, 230), font_path: str=BASE_FONT_PATH):
        """
        Initializes the dropdown component.

        Args:
            options (list): List of string options of the dropdown.
            fract_position (tuple): Fractional position in [0,1] of the component (x, y).
            fract_size (tuple): Fractional size in [0,1] of the dropdown WHEN CLOSED.
            open_upward (bool): If True, dropdown opens upward; if False, opens downward.
            bg_color (tuple): Background color of the dropdown in RGB format.
            text_color (tuple): Color of the text in RGB format.
            font_path (str): File path to the font to be used.
        """
        super().__init__(fract_position, fract_size)
        self.options = options
        self.font = SmartFont(fract_font_size=self.f_size[0]*0.5,font_path=font_path)

        self.font.f_font_size = self.f_size[0]*0.4 # Font size half of height of box
        self.selected_index = 0


        self.open_upward = open_upward
        
        # Colors
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_color = tuple(min(bg+30,255) for bg in bg_color) 
        self.hover_color = tuple(min(bg+20,255) for bg in bg_color)
        self.selected_color = tuple(255-bg for bg in bg_color)  # Invert background color for selected option
        self.selected_text_color = tuple(255-tc for tc in text_color)  # Invert text color for selected option

        self.border_size_fraction = 0.07  # Border size as a fraction of the dropdown height
        
        # State
        self.is_open = False
        self.hovered_index = -1
        
        # Rendered surfaces
        self.dropdown_surface = None
        self.option_surfaces = []
        self.dropdown_rect = None
        self.option_rects = []

    @property
    def selected(self):
        """
        Returns the currently selected option.
        """
        return self.options[self.selected_index]

    @selected.setter
    def selected(self, value):
        """
        Sets the selected option by value.
        """
        if value in self.options:
            self.selected_index = self.options.index(value)
        else:
            print(f"Warning: Value '{value}' not in options. Cannot set selected index.")
        
    def render(self):
        """
        Renders the dropdown component with correct font size and positioning.
        """
        self.font.sH = self.sH
        bg_color = self.hover_color if (self.hovered_index == -2) else self.bg_color
        text = self.selected
        if(self.is_open):
            text="x"

        self.dropdown_surface = self._render_option_surface(text, bg_color)

        # Prepare option surfaces
        self.option_surfaces = []
        self.option_rects = []
        
        for i, option in enumerate(self.options):
            # Choose bg_color and text_color based on hover/selection
            bg_color = self.bg_color
            text_color = self.text_color
            if(i == self.hovered_index):
                bg_color = self.hover_color
            if(i == self.selected_index):
                bg_color = self.selected_color
                text_color = self.selected_text_color

            option_surface = self._render_option_surface(option, bg_color, text_color=text_color)
            self.option_surfaces.append(option_surface)
            
            # Calculate option rectangle position
            if self.open_upward:
                option_y = self.y - (i+1) * self.h
            else:
                option_y = self.y + (i+1) * self.h
            
            self.option_rects.append(pygame.Rect(self.x, option_y, self.w, self.h))
        
        # Main dropdown rectangle
        self.dropdown_rect = pygame.Rect(self.x, self.y, self.w, self.h)

    def _render_option_surface(self, option, bg_color, text_color=None) -> pygame.Surface:
        """
            Renders and returns a surface for a single dropdown option.

            Args:
                option (str): The text of the option.
                bg_color (tuple): Background color for the option surface.
                txt_color (tuple): Text color for the option. If None, uses self.text_color.
            Returns:
                pygame.Surface: Option box with text label
        """
        option_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        option_surface.fill(bg_color)
        pygame.draw.rect(option_surface, self.border_color, 
                        (0, 0, self.w, self.h), max(1,int(self.border_size_fraction * self.h)))
        
        # Render option text
        text_surface = self.font.render(option, text_color if text_color else self.text_color)
        text_rect = text_surface.get_rect()
        text_rect.center = (self.w / 2, self.h / 2)
        option_surface.blit(text_surface, dest=text_rect)

        return option_surface

    def draw(self, screen: pygame.Surface) -> pygame.Surface:
        """
        Draws the dropdown component to the screen.
        """        
        # Draw main dropdown button
        screen.blit(self.dropdown_surface, (self.x, self.y))
        
        # Draw options if dropdown is open
        if self.is_open and self.option_surfaces:
            for i, (option_surface, option_rect) in enumerate(zip(self.option_surfaces, self.option_rects)):
                screen.blit(option_surface, (option_rect.x, option_rect.y))
        
        return screen

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handles events for the dropdown component.
        
        Args:
            event (pygame.event.Event): The event to handle.
        
        Returns:
            bool: True if the component state changed, False otherwise.
        """
        changed = False
        rerender = False
        if not self.dropdown_surface:
            return False  # No main surface means not rendered yet
    
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                
                # Check if clicking on main dropdown button
                if self.dropdown_rect.collidepoint(mouse_pos):
                    self.is_open = not self.is_open
                    rerender = True
                elif self.is_open:
                    clicked_inside=False
                    for i, option_rect in enumerate(self.option_rects):
                        if option_rect.collidepoint(mouse_pos):
                            self.selected_index = i
                            self.is_open = False
                            self.hovered_index = -1
                            changed = True
                            clicked_inside = True
                            break
                    if not clicked_inside:
                        self.is_open = False
                        self.hovered_index = -1
            
                    rerender = True
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            old_hovered = self.hovered_index      
            self.hovered_index = -1  

            if self.dropdown_rect.collidepoint(mouse_pos):
                self.hovered_index = -2 # Special value just to avoid yet another boolean
            elif self.is_open:
                for i, option_rect in enumerate(self.option_rects):
                    if option_rect.collidepoint(mouse_pos):
                        self.hovered_index = i
                        break
            
            # Re-render if hover state changed
            if old_hovered != self.hovered_index:
                rerender = True

        if(rerender):
            self.render()  # Re-render to update surfaces
    
        return changed