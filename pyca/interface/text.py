import pygame
import json

class TextBlock:
    """
    Represents a block of text to be displayed in the UI.
    """
    def __init__(self, text, position, color, font, y_offset = 0):
        self.text = text
        self.position = position  # "up_sx", "below_sx", "up_dx", "below_dx"
        self.color = color
        self.font = font
        self.y_offset = y_offset  # Will be calculated during layout

class DropdownMenu:
    def __init__(self, screen, width, height, font, options, default_option=None, margin=30):
        self.screen = screen
        self.width = width
        self.height = height
        self.margin = margin
        self.font = font
        self.options = options
        self.option_list = list(options.keys())
        self.active = False
        self.current_option = default_option if default_option else self.option_list[0]
        self.hover_index = -1
        self.label = "Select Automaton"
        self.selected_color = (230, 230, 230)  # White color for selected option
        self.normal_color = (50, 50, 50)       # Normal background color
        self.hover_color = (70, 70, 70)        # Hover background color
        
        # Calculate position based on screen size
        self.update_position()
        self.resize(width, height, margin)  # Use new resize method for initialization
        
    def update_position(self):
        screen_width, screen_height = self.screen.get_size()
        # Position at bottom right with margin
        x = screen_width - self.width - self.margin
        y = screen_height - self.height - self.margin
        
        # Update main rect
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
        # Update option rects
        self.option_rects = []
        for i in range(len(self.option_list)):
            # Options appear above the main button
            option_rect = pygame.Rect(x, y - (len(self.option_list) - i) * self.height, 
                                    self.width, self.height)
            self.option_rects.append(option_rect)
    
    def draw(self, screen, display_text=True):
        # Draw label
        if(display_text):
            label_surface = self.font.render(self.label, True, (230, 230, 230))
            label_rect = label_surface.get_rect(bottomright=(self.rect.right, self.rect.top - 5))
            # Draw label background
            padding = 5
            background_rect = pygame.Rect(
                label_rect.x - padding,
                label_rect.y - padding,
                label_rect.width + (padding * 2),
                label_rect.height + (padding * 2)
            )
            pygame.draw.rect(screen, (0, 0, 0), background_rect)
            screen.blit(label_surface, label_rect)
        
        # Draw main button with selected color
        pygame.draw.rect(screen, self.selected_color, self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
        
        # Draw current option in black (for contrast with white background)
        text_surface = self.font.render(self.current_option, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
        # Draw dropdown if active
        if self.active:
            for i, (option, rect) in enumerate(zip(self.option_list, self.option_rects)):
                # Set color based on hover and whether option is selected
                if i == self.hover_index:
                    color = self.hover_color
                elif option == self.current_option:
                    color = self.selected_color
                else:
                    color = self.normal_color
                    
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 2)
                
                # Use black text for selected option, white for others
                text_color = (0, 0, 0) if option == self.current_option else (230, 230, 230)
                text_surface = self.font.render(option, True, text_color)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if self.rect.collidepoint(event.pos):
                    self.active = not self.active
                elif self.active:
                    for i, rect in enumerate(self.option_rects):
                        if rect.collidepoint(event.pos):
                            self.current_option = self.option_list[i]  # Use option_list instead of options
                            self.active = False
                            return True  # Option was selected
                    self.active = False
        elif event.type == pygame.MOUSEMOTION and self.active:
            self.hover_index = -1
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(event.pos):
                    self.hover_index = i
                    break
        
        return False  # No option was selected

    def resize(self, width, height, margin, font=None):
        self.width = width
        self.height = height
        self.margin = margin
        if font is not None:
            self.font = font
        self.update_position()

def calculate_wrapped_height(text, font, screen_width, position):
    # Calculate max width based on position
    if position in ["up_sx", "below_sx"]:
        max_width = screen_width // 2 - 20
    else:
        max_width = 90

    total_lines = 0
    # Split text into initial lines (explicit line breaks)
    text_lines = text.splitlines()
    
    for text_line in text_lines:
        words = text_line.split()
        if not words:
            total_lines += 1
            continue
            
        current_width = 0
        for word in words:
            word_surface = font.render(word + ' ', True, (0,0,0))  # Color doesn't matter for size calc
            word_width = word_surface.get_width()
            
            if current_width + word_width >= max_width:
                total_lines += 1
                current_width = word_width
            else:
                if current_width == 0:
                    total_lines += 1
                current_width += word_width
    
    return total_lines * font.get_height() + 10  # +10 for padding

def render_text_blocks(screen, blocks):
    # Initialize position trackers
    up_sx_y = 10
    up_dx_y = 10
    below_dx_y = screen.get_height() - 120  # Initial position for below_dx
    
    # First calculate total height needed for below_sx blocks
    total_below_height = 0
    for block in blocks:
        if block.position == "below_sx":
            block_height = calculate_wrapped_height(block.text, block.font, screen.get_width(), block.position)
            total_below_height += block_height + 5  # +5 for spacing between blocks
    
    # Set starting y position for below_sx blocks
    below_sx_y = screen.get_height() - total_below_height - 20  # 20px padding from bottom
    
    # First pass: calculate y_offsets and render
    for block in blocks:
        if block.position == "up_sx":
            block.y_offset = up_sx_y
            height = blit_text(screen, block.text, block.position, block.font, block.color, y_offset=up_sx_y)
            up_sx_y += height + 5
        elif block.position == "below_sx":
            block.y_offset = below_sx_y
            height = blit_text(screen, block.text, block.position, block.font, block.color, y_offset=below_sx_y)
            below_sx_y += height + 5
        elif block.position == "up_dx":
            block.y_offset = up_dx_y
            height = blit_text(screen, block.text, block.position, block.font, block.color, y_offset=up_dx_y)
            up_dx_y += height + 5
        elif block.position == "below_dx":
            block.y_offset = below_dx_y
            height = blit_text(screen, block.text, block.position, block.font, block.color, y_offset=below_dx_y)
            below_dx_y += height + 5

def blit_text(screen, text, position, font, color, y_offset=None):
    if position == "up_sx":
        x = 10
        y = y_offset if y_offset is not None else 10
        max_width = screen.get_width() // 2 - 20
    elif position == "below_sx":
        x = 10
        y = y_offset if y_offset is not None else (screen.get_height() - 100)
        max_width = screen.get_width() // 2 - 20
    elif position == "up_dx":
        x = screen.get_width() - 100
        y = y_offset if y_offset is not None else 10
        max_width = 90
    elif position == "below_dx":
        x = screen.get_width() - 100
        y = y_offset if y_offset is not None else (screen.get_height() - 100)
        max_width = 90

    # Split text into lines first (preserve explicit line breaks)
    text_lines = text.splitlines()
    
    rendered_lines = []
    for text_line in text_lines:
        words = text_line.split(' ')
        if not words:
            rendered_lines.append('')
            continue
            
        current_line = []
        current_width = 0
        
        for word in words:
            word_surface = font.render(word + ' ', True, color)
            word_width = word_surface.get_width()
            
            if current_width + word_width >= max_width:
                if current_line:
                    rendered_lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
                else:
                    rendered_lines.append(word)
                    current_line = []
                    current_width = 0
            else:
                current_line.append(word)
                current_width += word_width
                
        if current_line:
            rendered_lines.append(' '.join(current_line))
    
    # Calculate total height and width for background
    total_height = len(rendered_lines) * font.get_height()
    max_line_width = 0
    for line in rendered_lines:
        if line:
            line_width = font.render(line, True, color).get_width()
            max_line_width = max(max_line_width, line_width)
    
    # Draw background
    padding = 5
    background_rect = pygame.Rect(
        x - padding, 
        y - padding,
        max_line_width + (padding * 2),
        total_height + (padding * 2)
    )
    pygame.draw.rect(screen, (0, 0, 0), background_rect)  # Black background only
    
    # Render all lines
    for i, line in enumerate(rendered_lines):
        if line:
            text_surface = font.render(line, True, color)
            screen.blit(text_surface, (x, y + i * font.get_height()))
    
    return total_height

class InputField:
    def __init__(self, screen, width, height, font, label, initial_value, margin=30, index=0, y=70):
        self.screen = screen
        self.width = width
        self.height = height
        self.font = font
        self.text = str(initial_value)
        self.stored_value = initial_value
        self.label = label
        self.active = False
        self.color = (50, 50, 50)
        self.margin = margin
        self.label_color = (230, 230, 230)
        self.index = index
        self.y = y
        self.update_position()
        self.resize(width, height, margin)  # Use new resize method for initialization
    
    def update_position(self):
        screen_width = self.screen.get_width()
        # Position at top right with margin, stacked vertically
        x = screen_width - self.width - self.margin - (self.index * (self.width + 20))  # Stack horizontally from right
        y = self.y  # Fixed vertical position below FPS text
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
    def draw(self):
        # Draw label
        label_surface = self.font.render(self.label, True, self.label_color)
        label_rect = label_surface.get_rect(bottomright=(self.rect.right, self.rect.top - 5))
        # Draw label background
        padding = 5
        background_rect = pygame.Rect(
            label_rect.x - padding,
            label_rect.y - padding,
            label_rect.width + (padding * 2),
            label_rect.height + (padding * 2)
        )
        pygame.draw.rect(self.screen, (0, 0, 0), background_rect)
        self.screen.blit(label_surface, label_rect)
        
        # Draw input box
        color = (100, 100, 100) if self.active else (50, 50, 50)
        pygame.draw.rect(self.screen, color, self.rect)
        pygame.draw.rect(self.screen, (100, 100, 100), self.rect, 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, (230, 230, 230))
        text_rect = text_surface.get_rect(center=self.rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            was_active = self.active
            # If clicking on the field
            if self.rect.collidepoint(event.pos):
                if not self.active:  # Only clear if newly activated
                    self.text = ""
                self.active = True
            else:
                self.active = False
                if was_active:  # If we're clicking away from an active field
                    new_value = self.get_value()
                    if new_value and new_value > 0:
                        self.stored_value = new_value
                        return True  # Signal that value has changed
                    else:
                        self.text = str(self.stored_value)  # Restore previous value if invalid
        
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                new_value = self.get_value()
                if new_value and new_value > 0:
                    self.stored_value = new_value
                else:
                    self.text = str(self.stored_value)  # Restore previous value if invalid
                self.active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # Only allow numbers
                if event.unicode.isnumeric():
                    self.text += event.unicode
        return False
    
    def get_value(self):
        try:
            return int(self.text)
        except ValueError:
            return None

    def resize(self, width, height, margin, font=None):
        self.width = width
        self.height = height
        self.margin = margin
        if font is not None:
            self.font = font
        self.update_position()
