import torch
import numpy as np
from PIL import Image

def add_gaussian_noise(img_tensor: torch.Tensor, mean: float = 0.0, std: float = 0.1) -> torch.Tensor:
    """
    Adds additive Gaussian noise to an image tensor [C, H, W] or [B, C, H, W] in range [0, 1].
    """
    noise = torch.randn_like(img_tensor) * std + mean
    noisy_img = torch.clamp(img_tensor + noise, 0.0, 1.0)
    return noisy_img

def add_poisson_noise(img_tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Simulates Poisson (shot) noise by scaling intensity, applying Poisson distribution, and rescaling.
    """
    # Scale image to photon counts (e.g. max 255 * scale)
    max_val = 255.0 * max(scale, 0.1)
    vals = len(torch.unique(img_tensor))
    vals = 2 ** np.ceil(np.log2(vals)) if vals > 0 else 256
    scaled = img_tensor * max_val
    noisy = torch.poisson(torch.clamp(scaled, min=0.0)) / max_val
    return torch.clamp(noisy, 0.0, 1.0)

def add_salt_and_pepper_noise(img_tensor: torch.Tensor, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> torch.Tensor:
    """
    Adds Salt & Pepper noise to an image tensor.
    """
    noisy_img = img_tensor.clone()
    
    # Salt noise (white pixels)
    salt_mask = torch.rand_like(noisy_img[:1]) < salt_prob
    noisy_img = torch.where(salt_mask.expand_as(noisy_img), torch.tensor(1.0, device=img_tensor.device), noisy_img)
    
    # Pepper noise (black pixels)
    pepper_mask = torch.rand_like(noisy_img[:1]) < pepper_prob
    noisy_img = torch.where(pepper_mask.expand_as(noisy_img), torch.tensor(0.0, device=img_tensor.device), noisy_img)
    
    return noisy_img

def apply_noise(img_tensor: torch.Tensor, noise_type: str = "gaussian", severity: float = 0.1) -> torch.Tensor:
    """
    Unified function to apply requested noise type with severity in [0.0, 1.0].
    """
    if noise_type.lower() == "gaussian":
        std = severity * 0.3  # Map 0..1 to 0..0.3 std
        return add_gaussian_noise(img_tensor, std=std)
    elif noise_type.lower() == "poisson":
        scale = max(0.1, 1.0 - severity * 0.8)
        return add_poisson_noise(img_tensor, scale=scale)
    elif noise_type.lower() in ["salt_pepper", "salt_and_pepper", "sp"]:
        prob = severity * 0.05
        return add_salt_and_pepper_noise(img_tensor, salt_prob=prob, pepper_prob=prob)
    elif noise_type.lower() == "none":
        return img_tensor
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
