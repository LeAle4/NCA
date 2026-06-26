"""
patterns.py — Generation of starting conditions (tensors) for testing
and training PDE/NCA models.
"""

import torch
import numpy as np

from config import DEVICE, STD_DTYPE

def make_random_noise(shape, scale=0.1, device=DEVICE, dtype=STD_DTYPE):
    """Generates standard normally distributed random noise."""
    return scale * torch.randn(*shape, device=device, dtype=dtype)

def make_gaussian_bump(shape, sigma=5.0, amplitude=1.0, center=None, device=DEVICE, dtype=STD_DTYPE):
    """
    Generates a single Gaussian bump in the center of the grid (or specified center).
    
    shape: (B, C, H, W)
    """
    B, C, H, W = shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    
    if center is None:
        cy, cx = H / 2.0, W / 2.0
    else:
        cy, cx = center

    r2 = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
    bump = amplitude * torch.exp(-r2 / (2.0 * sigma ** 2))
    
    # Broadcast to shape (B, C, H, W)
    return bump.view(1, 1, H, W).repeat(B, C, 1, 1)

def make_multiple_gaussian_bumps(shape, num_bumps=5, sigma_range=(3.0, 7.0), amp_range=(0.5, 1.5), device=DEVICE, dtype=STD_DTYPE):
    """Generates a tensor with multiple randomly placed Gaussian bumps."""
    B, C, H, W = shape
    out = torch.zeros(shape, device=device, dtype=dtype)
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    
    for b in range(B):
        for c in range(C):
            for _ in range(num_bumps):
                cy = np.random.uniform(0.1 * H, 0.9 * H)
                cx = np.random.uniform(0.1 * W, 0.9 * W)
                sigma = np.random.uniform(*sigma_range)
                amplitude = np.random.uniform(*amp_range)
                
                r2 = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
                bump = amplitude * torch.exp(-r2 / (2.0 * sigma ** 2))
                out[b, c] += bump
                
    return out

def make_cos_sine_waves(shape, frequency=2.0, amplitude=1.0, mode="both", device=DEVICE, dtype=STD_DTYPE):
    """
    Generates structured periodic wave patterns.
    
    mode: "horizontal", "vertical", "diagonal", "both"
    """
    B, C, H, W = shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    
    # Normalized coordinates in [0, 2*pi]
    y_scaled = 2 * np.pi * grid_y / H
    x_scaled = 2 * np.pi * grid_x / W
    
    if mode == "horizontal":
        pattern = amplitude * torch.cos(frequency * x_scaled)
    elif mode == "vertical":
        pattern = amplitude * torch.cos(frequency * y_scaled)
    elif mode == "diagonal":
        pattern = amplitude * torch.cos(frequency * (x_scaled + y_scaled) / np.sqrt(2))
    else:  # "both" (checkerboard grid)
        pattern = amplitude * torch.cos(frequency * x_scaled) * torch.cos(frequency * y_scaled)
        
    return pattern.view(1, 1, H, W).repeat(B, C, 1, 1)

def make_step_function(shape, value_inside=1.0, value_outside=0.0, radius=0.25, center=None, device=DEVICE, dtype=STD_DTYPE):
    """
    Generates a step function (circular block in the middle).
    
    radius: fraction of the minimum dimension [0, 0.5]
    """
    B, C, H, W = shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    
    if center is None:
        cy, cx = H / 2.0, W / 2.0
    else:
        cy, cx = center
        
    r_pixel = radius * min(H, W)
    r = torch.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)
    
    pattern = torch.where(r <= r_pixel, torch.tensor(value_inside, device=device, dtype=dtype), torch.tensor(value_outside, device=device, dtype=dtype))
    return pattern.view(1, 1, H, W).repeat(B, C, 1, 1)

def make_striped_pattern(shape, stripe_width=4, orientation="horizontal", device=DEVICE, dtype=STD_DTYPE):
    """
    Generates a binary striped pattern.
    
    stripe_width: width of stripes in pixels
    orientation: "horizontal" or "vertical"
    """
    B, C, H, W = shape
    
    if orientation == "horizontal":
        stripes = (torch.arange(H, device=device, dtype=dtype) // stripe_width) % 2
        pattern = stripes.view(1, 1, H, 1).repeat(B, C, 1, W)
    else:
        stripes = (torch.arange(W, device=device, dtype=dtype) // stripe_width) % 2
        pattern = stripes.view(1, 1, 1, W).repeat(B, C, H, 1)
        
    return pattern
