from .TextBox import TextBox


class TextLabel(TextBox):
    """
    Represents a label using TextBox, easier to deal with than UILabel.
    """

    def __init__(self, text, manager, parent=None, rel_pos=(0,0), rel_size=(-1,0.1), font_size=12):
        """
        Initializes the text label component.

        Args:
            text (str): The text to display.
            manager: pygame-gui UIManager instance.
            parent: parent BaseComponent if any. All relative quantities are relative to this container.
            rel_pos (tuple): Fractional position in [0,1] of the component (x, y).
            rel_size (tuple): Fractional size in [0,1] of the component (height, width). Height can be -1 for auto.
            font_size: Base is 12, and changes are proportional based on that
            z_pos: Z-position for rendering order. Higher values are rendered on top.
        """
        class_id = f"@better_label"
        super().__init__(text=text, font_size=font_size, manager=manager, parent=parent, rel_pos=rel_pos, rel_size=rel_size, theme_class=class_id)

