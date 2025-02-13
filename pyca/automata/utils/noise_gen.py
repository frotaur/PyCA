import torch
import numpy as np, math
from pyperlin import FractalPerlin2D

def perlin(shape:tuple, wavelengths:tuple, num_channels=3, black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.

        Args:
        shape : (B,H,W) tuple or array, size of tensor
        wavelength : int, wavelength of the noise in pixels
        black_prop : percentage of black (0) regions. Defaut is .5=50% of black.
        device : device for returned tensor

        Returns :
        (B,3,H,W) torch tensor of perlin noise.
    """    
    B,H,W = tuple(shape)
    lams = tuple(int(wave) for wave in wavelengths)
    # Extend image so that its integer wavelengths of noise
    W_new=int(W+(lams[0]-W%lams[0]))
    H_new=int(H+(lams[1]-H%lams[1]))
    frequency = [H_new//lams[0],W_new//lams[1]]
    gen = torch.Generator(device=device) # for GPU acceleration
    gen.seed()
    # Strange 1/0.7053 factor to get images noise in range (-1,1), quirk of implementation I think...
    fp = FractalPerlin2D((B*num_channels,H_new,W_new), [frequency], [1/0.7053], generator=gen)()[:,:H,:W].reshape(B,num_channels,H,W) # (B,C,H,W) noise)

    return torch.clamp((fp+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1)

def perlin_fractal(shape:tuple, max_wavelength:int, persistence=0.5,num_channels=3,black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.
    """
    max_num = min(6,int(math.log2(max_wavelength)))
    normalization = float(sum([persistence**(i+1) for i in range(max_num)]))
    return 1./normalization*sum([persistence**(i+1)*perlin(shape,[int(2**(-i)*max_wavelength)]*2,black_prop=black_prop,num_channels=num_channels,device=device) for i in range(max_num)])

if __name__=='__main__':
    from time import time
    from torchenhanced.util import saveTensImage
    size = 400,400
    device = 'cpu'
    waves = np.array([60,60])

    t0 = time()
    tens = perlin_fractal((1,*size),waves[0],black_prop=0.5,device='cpu')
    saveTensImage(tens.cpu(),'.','fast')
    print('perlin fast : ', time()-t0)

    t0 = time()
    for i in range(1):
        tens = perlin_fractal(size,waves[0],black_prop=0.5)
    saveTensImage(tens,'.','slow')
    print('perlin : ', time()-t0)