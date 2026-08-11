import torch
import numpy as np
from .noise import add_poisson_noise

def apply_gamma_darkening(img_tensor: torch.Tensor, gamma: float = 2.5) -> torch.Tensor:
    """
    Applies gamma transformation I_out = I_in ^ gamma (gamma > 1 darkens image).
    """
    return torch.pow(torch.clamp(img_tensor, 1e-6, 1.0), gamma)

def apply_illumination_scaling(img_tensor: torch.Tensor, scale: float = 0.3) -> torch.Tensor:
    """
    Linearly scales image brightness.
    """
    return torch.clamp(img_tensor * scale, 0.0, 1.0)

def apply_low_light(img_tensor: torch.Tensor, severity: float = 0.5, add_noise: bool = True) -> torch.Tensor:
    """
    Unified function for simulating realistic low-light degradation:
    - Reduced illumination (brightness dimming)
    - Gamma attenuation
    - Optional shot/Poisson noise in dark regions
    """
    if severity <= 0.0:
        return img_tensor
        
    # Scale illumination between 0.1 (darkest) and 0.6
    brightness_scale = 0.6 - (severity * 0.5)
    darkened = apply_illumination_scaling(img_tensor, scale=brightness_scale)
    
    # Apply gamma non-linearity (gamma 1.5 to 4.0)
    gamma = 1.5 + (severity * 2.5)
    darkened = apply_gamma_darkening(darkened, gamma=gamma)
    
    if add_noise and severity > 0.1:
        darkened = add_poisson_noise(darkened, scale=brightness_scale * 2.0)
        
    return torch.clamp(darkened, 0.0, 1.0)
