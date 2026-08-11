import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2

def apply_gaussian_blur(img_tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """
    Applies Gaussian Blur using torchvision functional transforms.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(img_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

def apply_motion_blur(img_tensor: torch.Tensor, size: int = 9, angle: float = 45.0) -> torch.Tensor:
    """
    Applies Motion Blur using directional convolution kernel.
    """
    if size <= 1:
        return img_tensor

    # Create motion blur kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, :] = 1.0
    
    # Rotate kernel
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel = kernel / np.sum(kernel)
    
    kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(img_tensor.device)
    
    # Apply depthwise convolution per channel
    channels = img_tensor.shape[-3]
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
    
    is_batched = (img_tensor.dim() == 4)
    if not is_batched:
        img_tensor = img_tensor.unsqueeze(0)
        
    padded = torch.nn.functional.pad(img_tensor, (center, center, center, center), mode='reflect')
    blurred = torch.nn.functional.conv2d(padded, kernel_tensor, groups=channels)
    
    if not is_batched:
        blurred = blurred.squeeze(0)
        
    return torch.clamp(blurred, 0.0, 1.0)

def apply_defocus_blur(img_tensor: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """
    Applies Defocus (disk) blur.
    """
    if radius <= 1:
        return img_tensor

    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()

    kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(img_tensor.device)
    channels = img_tensor.shape[-3]
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)

    is_batched = (img_tensor.dim() == 4)
    if not is_batched:
        img_tensor = img_tensor.unsqueeze(0)

    padded = torch.nn.functional.pad(img_tensor, (radius, radius, radius, radius), mode='reflect')
    blurred = torch.nn.functional.conv2d(padded, kernel_tensor, groups=channels)

    if not is_batched:
        blurred = blurred.squeeze(0)

    return torch.clamp(blurred, 0.0, 1.0)

def apply_blur(img_tensor: torch.Tensor, blur_type: str = "gaussian", severity: float = 0.5) -> torch.Tensor:
    """
    Unified function to apply requested blur type with severity in [0.0, 1.0].
    """
    if blur_type.lower() == "gaussian":
        k_size = int(3 + severity * 10)
        sigma = 0.5 + severity * 3.0
        return apply_gaussian_blur(img_tensor, kernel_size=k_size, sigma=sigma)
    elif blur_type.lower() == "motion":
        size = int(3 + severity * 12)
        angle = severity * 180.0
        return apply_motion_blur(img_tensor, size=size, angle=angle)
    elif blur_type.lower() == "defocus":
        radius = int(1 + severity * 6)
        return apply_defocus_blur(img_tensor, radius=radius)
    elif blur_type.lower() == "none":
        return img_tensor
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")
