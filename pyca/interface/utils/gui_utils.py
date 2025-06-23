from ..text import TextBlock

def make_text_blocks(description, help_text, std_help, font, font_title):
    text_blocks = [
        TextBlock(description, "up_sx", (74, 101, 176), font_title),
        TextBlock("\n", "up_sx", (230, 230, 230), font)
    ]
    for section in std_help['sections']:
        text_blocks.append(TextBlock(section["title"], "up_sx", (230, 89, 89), font))
        for command, description in section["commands"].items():
            text_blocks.append(TextBlock(f"{command} -> {description}", "up_sx", (230, 230, 230), font))
        text_blocks.append(TextBlock("\n", "up_sx", (230, 230, 230), font))
    text_blocks.append(TextBlock("Automaton controls", "below_sx", (230, 89, 89), font))
    text_blocks.append(TextBlock(help_text, "below_sx", (230, 230, 230), font))
    return text_blocks
