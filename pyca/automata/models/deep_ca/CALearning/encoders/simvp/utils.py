from torchenhanced import DevModule
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_



def sampling_generator(N, reverse=False):
    """
        Generate a list of bools, wether to downsample or not
    """
    samplings = [False, True] * (N // 2)+[False] * (N % 2)  # Ensure even length
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class TorusConv2d(DevModule):
    """
        Basically just Conv2d, with extra torus padding
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 toruspad=True,
                 device='cpu'):
        """
        Initializes a basic convolutional layer with optional activation normalization and toroidal padding.
            Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple): Stride of the convolution.
            padding (int or tuple): Padding added to both sides of the input.
            dilation (int or tuple): Spacing between kernel elements.
            toruspad (bool): Whether to use toroidal padding.
        """
        super().__init__(device=device)
        self.padding=padding
        self.toruspad = toruspad
        if(toruspad):
            padding=0
        

        self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if(self.toruspad):
            x = F.pad(x, pad=[self.padding for _ in range(4)],mode='circular') # Torus Pad
        
        return self.conv(x)
