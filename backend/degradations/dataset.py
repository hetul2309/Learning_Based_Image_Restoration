import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .noise import apply_noise
from .blur import apply_blur
from .low_light import apply_low_light

class RestorationDataset(Dataset):
    """
    PyTorch Dataset that loads clean images and applies configurable,
    on-the-fly degradations (noise, blur, low-light) returning (degraded, clean) image tensor pairs.
    """
    def __init__(self, root_dir, image_size=(128, 128), degradation_config=None, transform=None):
        """
        root_dir: directory containing clean PNG/JPG images
        degradation_config: dict specifying probability & parameters for degradations
          e.g. {
            'noise': {'types': ['gaussian', 'poisson', 'salt_pepper'], 'prob': 0.8, 'severity_range': (0.1, 0.4)},
            'blur': {'types': ['gaussian', 'motion', 'defocus'], 'prob': 0.6, 'severity_range': (0.2, 0.6)},
            'low_light': {'prob': 0.5, 'severity_range': (0.3, 0.7)}
          }
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.degradation_config = degradation_config or {
            'noise': {'types': ['gaussian', 'poisson', 'salt_pepper'], 'prob': 0.7, 'severity_range': (0.05, 0.3)},
            'blur': {'types': ['gaussian', 'motion', 'defocus'], 'prob': 0.5, 'severity_range': (0.1, 0.5)},
            'low_light': {'prob': 0.4, 'severity_range': (0.2, 0.6)}
        }
        
        self.image_paths = []
        if os.path.exists(root_dir):
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
                
        self.to_tensor = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def apply_degradations(self, clean_tensor):
        degraded = clean_tensor.clone()
        
        # Apply Low-light
        ll_cfg = self.degradation_config.get('low_light', {})
        if torch.rand(1).item() < ll_cfg.get('prob', 0.0):
            sev_min, sev_max = ll_cfg.get('severity_range', (0.2, 0.6))
            severity = torch.empty(1).uniform_(sev_min, sev_max).item()
            degraded = apply_low_light(degraded, severity=severity)
            
        # Apply Blur
        blur_cfg = self.degradation_config.get('blur', {})
        if torch.rand(1).item() < blur_cfg.get('prob', 0.0):
            blur_type = blur_cfg.get('types', ['gaussian'])[int(torch.randint(0, len(blur_cfg.get('types', ['gaussian'])), (1,)).item())]
            sev_min, sev_max = blur_cfg.get('severity_range', (0.1, 0.5))
            severity = torch.empty(1).uniform_(sev_min, sev_max).item()
            degraded = apply_blur(degraded, blur_type=blur_type, severity=severity)

        # Apply Noise
        noise_cfg = self.degradation_config.get('noise', {})
        if torch.rand(1).item() < noise_cfg.get('prob', 0.0):
            noise_type = noise_cfg.get('types', ['gaussian'])[int(torch.randint(0, len(noise_cfg.get('types', ['gaussian'])), (1,)).item())]
            sev_min, sev_max = noise_cfg.get('severity_range', (0.05, 0.3))
            severity = torch.empty(1).uniform_(sev_min, sev_max).item()
            degraded = apply_noise(degraded, noise_type=noise_type, severity=severity)

        return degraded

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img_rgb = img.convert('RGB')
            clean_tensor = self.to_tensor(img_rgb)
            
        degraded_tensor = self.apply_degradations(clean_tensor)
        return degraded_tensor, clean_tensor
