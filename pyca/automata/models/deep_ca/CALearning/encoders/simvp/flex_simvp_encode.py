from ..encoder import Encoder
from .utils import sampling_generator, TorusConv2d
import torch.nn as nn
from torchenhanced import ConfigModule

class FlexSimVPEncoder(Encoder):
    """
        Re-implemented SimVP Encoder with FlexiGPT-compatible forward function.
    """

    def __init__(self, C_in:int, C_hid:int, N_E:int, kernel_size:int=3, device='cpu'):
        """
        Args:
            C_in : number of input channels
            C_hid : number of hidden channels
            N_E : number of layers
            kernel_size : size of the convolution kernel (3 by default)
            device : device to run the model on
        """
        super().__init__(device=device)

        self.C_in = C_in
        self.C_hid = C_hid
        self.N_E = N_E
        self.kernel_size = kernel_size


        down_sample_sequence = sampling_generator(N_E)
        self.enc = nn.Sequential(
            *[self._conv_layer(C_in if i == 0 else C_hid, C_hid, downsampling=down_sample_sequence[i], device=self.device) for i in range(N_E)]
        )


    def _conv_layer(self,c_in:int, c_out:int,downsampling,device='cpu'):
        """
            Returns conv layer followed by group norm and activation
        """
        padding= (self.kernel_size - 1) // 2
        stride=1
        if(downsampling):
            stride = 2

        conv = TorusConv2d(in_channels=c_in, out_channels=c_out, kernel_size=self.kernel_size, stride=stride, padding=padding, device=device)
        group_norm = nn.GroupNorm(num_groups=2, num_channels=c_out)
        act = nn.SiLU()
        return nn.Sequential(conv, group_norm, act)

    def forward(self, x):
        """
            Encodes the input video batch

            params :
            x : tensor of shape (B, C, H, W)

            returns :
            latent : encoded batch of images (B, C_hid, H_, W_) H_ = H/2**(N_E/2) (same W_)
            enc1 : first layer encoding, to use for skip connection (B, C_hid, H, W)
        """
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)

        return latent, enc1

class FlexSimVPDecoder(ConfigModule):
    """
        Re-implemented SimVP Decoder with FlexiGPT-compatible forward function.
    """
    def __init__(self, C_hid:int, C_out:int, N_D:int, kernel_size:int=3, skip=False, device='cpu'):
        """
        Args:
            C_hid : number of hidden channels
            C_out : number of output channels
            N_D : number of layers
            kernel_size : size of the convolution kernel (3 by default)
            skip : whether to use skip connections with the encoder first layer
            device : device to run the model on
        """
        super().__init__(device)
        self.C_hid = C_hid
        self.C_out = C_out
        self.N_D = N_D
        self.kernel_size = kernel_size

        up_sample_sequence = sampling_generator(N_D, reverse=True)
        self.dec = nn.Sequential(
            *[self._conv_layer(C_hid,C_hid, upsampling=up_sample_sequence[i]) for i in range(N_D)]
        )

        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.skip = skip

    def _conv_layer(self, c_in:int, c_out:int, upsampling=False):
        """
            Returns appropriate conv layer for SimVP Decoder
        """
        padding = (self.kernel_size - 1) // 2
        if(upsampling):
            conv_c_out = c_out*4 # After pixel shuffle, will have c_out again but scaled by 2
        else:
            conv_c_out = c_out
        conv = TorusConv2d(in_channels=c_in, out_channels=conv_c_out, kernel_size=self.kernel_size, stride=1, padding=padding)
        if(upsampling):
            
            conv = nn.Sequential(
                conv,
                nn.PixelShuffle(2)  # Upsample by a factor of 2
            )


        group_norm = nn.GroupNorm(num_groups=2, num_channels=c_out)
        act = nn.SiLU()
        return nn.Sequential(conv, group_norm, act)

    def forward(self, latent, skips=None):
        """
            Decodes the latent batch

            params :
            latent : encoded batch of images (B, C_hid, H_, W_)
            skips : (B, C_hid, H, W) first layer encoding from the encoder, to use for skip connection

            returns :
            decoded : tensor of shape (B, C_out, H', W') H' = H_/2**(N_D/2) (same W')
        """
        for i in range(len(self.dec)-1):
            latent = self.dec[i](latent)

        if(self.skip and skips is not None):
            latent = self.dec[-1](latent + skips)
        else:
            latent = self.dec[-1](latent)
        
        return self.readout(latent) # (B, C_out, H', W') where H' = H_/2**(N_D/2) (same W')