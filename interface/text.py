import pygame
import json

class TextBlock:
    def __init__(self, text, position, color, font):
        self.text = text
        self.position = position  # "up_sx", "below_sx", etc.
        self.color = color
        self.font = font
        self.y_offset = 0  # Will be calculated during layout

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
    pygame.draw.rect(screen, (0, 0, 0), background_rect)  # Black background
    pygame.draw.rect(screen, (50, 50, 50), background_rect, 1)  # Gray border
    
    # Render all lines
    for i, line in enumerate(rendered_lines):
        if line:
            text_surface = font.render(line, True, color)
            screen.blit(text_surface, (x, y + i * font.get_height()))
    
    return total_height

def load_std_help() :
    with open('interface/std_help.json','r') as f:
        return json.load(f)