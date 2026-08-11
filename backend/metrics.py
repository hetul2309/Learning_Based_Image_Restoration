import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between two images [H, W, C] in range [0, 1].
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0  # Perfect match
    return float(10.0 * np.log10((max_val ** 2) / mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index Measure (SSIM) between two images [H, W, C].
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        
    # skimage ssim requires channel_axis for 3D images
    if img1.ndim == 3:
        score = ssim_fn(img1, img2, channel_axis=2, data_range=1.0)
    else:
        score = ssim_fn(img1, img2, data_range=1.0)
    return float(score)

def evaluate_metrics(clean_img: np.ndarray, degraded_img: np.ndarray, restored_img: np.ndarray) -> dict:
    """
    Computes comparative PSNR and SSIM metrics for Degraded vs Clean and Restored vs Clean.
    """
    psnr_degraded = calculate_psnr(clean_img, degraded_img)
    ssim_degraded = calculate_ssim(clean_img, degraded_img)
    
    psnr_restored = calculate_psnr(clean_img, restored_img)
    ssim_restored = calculate_ssim(clean_img, restored_img)
    
    return {
        "degraded": {
            "psnr": round(psnr_degraded, 2),
            "ssim": round(ssim_degraded, 4)
        },
        "restored": {
            "psnr": round(psnr_restored, 2),
            "ssim": round(ssim_restored, 4)
        },
        "gain": {
            "psnr_gain": round(psnr_restored - psnr_degraded, 2),
            "ssim_gain": round(ssim_restored - ssim_degraded, 4)
        }
    }
