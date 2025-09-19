import torch

def create_smooth_circular_mask(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Creates a smooth circular mask for the provided batched image tensor. Used to 'cut' the kernel in a circle.
    """
    H, W = tensor.shape[-2], tensor.shape[-1]
    center_y = (H - 1) / 2  # Allow fractional center for better smoothness
    center_x = (W - 1) / 2
    y = torch.linspace(0, H - 1, H, device=tensor.device).view(-1, 1)
    x = torch.linspace(0, W - 1, W, device=tensor.device).view(1, -1)
    distance = ((y - center_y) ** 2 + (x - center_x) ** 2).sqrt()
    smooth_transition = 0.5  # Define a region for the smooth transition (around the edge of the circle)
    mask = torch.clamp(1 - (distance - radius) / smooth_transition, 0, 1)
    masked_tensor = tensor * mask

    return masked_tensor
