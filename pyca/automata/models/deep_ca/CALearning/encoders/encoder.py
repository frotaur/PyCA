from torchenhanced import ConfigModule
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(ConfigModule):
    """
        Generic Encoder class, has methods to access latent-shapes.
    """
    def __init__(self,device='cpu'):
        super().__init__(device)

    @torch.no_grad()
    def get_lat_shape(self, img_shape):
        """
        Returns latent shape (C,H_,W_), to be used as input for decoder

        Args:
        img_shape : tuple (C,H,W), shpae of input images to encode
        """
        x = self.forward(torch.randn(1,*img_shape)) # (1, C_hid, H', W')

        if(isinstance(x, tuple)):
            x = x[0]
        return x.shape[1:]

    @torch.no_grad()
    def get_lat_chans(self, img_shape):
        """
        Returns latent channels, to be used as input for decoder

        Args:
        img_shape : tuple (C,H,W), shpae of input images to encode

        Returns:
        int : number of latent channels, i.e. E=C'*H'*W'
        """
        shapu = self.get_lat_shape(img_shape)

        product = 1
        for i in range(len(shapu)):
            product *= shapu[i]

        return product