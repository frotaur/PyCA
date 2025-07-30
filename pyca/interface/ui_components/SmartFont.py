import pygame

class SmartFont:
    """
        To handle fonts more gracefully, especially under resize.
    """
    def __init__(self, fract_font_size=1./20., font_path=None, font_name=None, base_sH=None):
        """
            Initializes SmartFont with font path or name, and fractional size. Note that the font
            requires the screen height to be set before it can be used for rendering. Either use
            a base_sH, or set self.sH later.

            Args:
                fract_font_size (float): Fractional size of the font. (in fraction of screen height)
                font_path (str, optional): Path to the font file. If None, uses default system font.
                font_name (str, optional): Name of the font. If provided, overrides font_path.
                base_sH (int, optional): Base screen height for calculating font size. If None, 
                    font will be unusable until sH is set.
        """
        self.font_path = font_path
        self.font_name = font_name
        self._f_size = fract_font_size
        
        if(base_sH is not None):
            self._size = int(base_sH * self._f_size)
            self._sH = base_sH
        else:
            self._size = None
        
        self.font = self._create_font()
    
    def _create_font(self):
        if(self._size is not None):
            if self.font_name:
                return pygame.font.SysFont(self.font_name, self._size)
            else:
                return pygame.font.Font(self.font_path, self._size)
        else:
            return None
        
    @property
    def sH(self):
        """
            Returns the screen height used for calculating font size.
        """
        return self._sH

    @sH.setter
    def sH(self, new_sH):
        """
            Sets the screen height and updates the font size accordingly.
        """
        self._size = int(new_sH * self._f_size)
        self._sH = new_sH
        self.font = self._create_font()
    
    @property
    def f_font_size(self):
        """
            Returns the fractional font size.
        """
        return self._f_size

    @f_font_size.setter
    def f_font_size(self, new_f_size):
        """
            Sets the fractional font size and updates the font size accordingly.
        """
        self._f_size = new_f_size
        if self.sH is not None:
            self._size = int(self.sH * self._f_size)
            self.font = self._create_font()
    
    @property
    def height(self):
        """
            Return the actual, set, font height (NOT FRACTIONAL!!)
        """
        return self.font.get_height() if self.font else 0
    
    @property
    def f_height(self):
        """
            Returns the fractional height of the font based on the screen height.
        """
        return self.height / self.sH if self.sH else 0