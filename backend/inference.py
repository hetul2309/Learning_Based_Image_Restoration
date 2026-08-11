import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CNNRestoration2D, ConvolutionalAutoencoder, RestorationGANGenerator, Restormer

MODEL_REGISTRY = {
    "cnn": {
        "name": "CNN (Deep 2D Convolutional Network)",
        "class": CNNRestoration2D,
        "weight_paths": [
            "backend/cnn/cnn_model.pth",
            "backend/weights/cnn_model.pth",
            "cnn/cnn_model.pth",
            "weights/cnn_model.pth"
        ],
        "description": "Deep 2D Convolutional Neural Network operating at full resolution with residual learning."
    },
    "autoencoder": {
        "name": "Convolutional Autoencoder",
        "class": ConvolutionalAutoencoder,
        "weight_paths": [
            "backend/autoencoder/autoencoder_model.pth",
            "backend/weights/autoencoder_model.pth",
            "autoencoder/autoencoder_model.pth",
            "weights/autoencoder_model.pth"
        ],
        "description": "Encoder-Decoder Autoencoder with spatial compression and latent bottleneck."
    },
    "gan": {
        "name": "GAN (Generative Adversarial Network)",
        "class": RestorationGANGenerator,
        "weight_paths": [
            "backend/gan/gan_generator.pth",
            "backend/weights/gan_generator.pth",
            "gan/gan_generator.pth",
            "weights/gan_generator.pth"
        ],
        "description": "Dense Residual Block GAN Generator with PatchGAN & VGG Perceptual Loss."
    },
    "transformer": {
        "name": "Transformer (Restormer)",
        "class": Restormer,
        "weight_paths": [
            "backend/transformer/transformer_model.pth",
            "backend/weights/transformer_model.pth",
            "transformer/transformer_model.pth",
            "weights/transformer_model.pth"
        ],
        "description": "Restormer Vision Transformer with Multi-Dconv Head Transposed Self-Attention."
    }
}

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
        print(f"[ModelManager] Operating on Device: {self.device}")

    def get_available_models(self):
        models_info = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for key, info in MODEL_REGISTRY.items():
            has_weights = False
            found_weight_path = None
            for wp in info["weight_paths"]:
                full_path = os.path.join(project_root, wp)
                if os.path.exists(full_path):
                    has_weights = True
                    found_weight_path = full_path
                    break
            
            models_info.append({
                "id": key,
                "name": info["name"],
                "description": info["description"],
                "has_weights": has_weights,
                "weight_path": found_weight_path
            })
        return models_info

    def get_model(self, model_id: str):
        if model_id not in MODEL_REGISTRY:
            # Fallback mapping for legacy names
            legacy_map = {
                "cnn_autoencoder": "cnn",
                "dae_vae": "autoencoder",
                "restoration_gan": "gan",
                "restormer": "transformer"
            }
            model_id = legacy_map.get(model_id, model_id)
            
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_id: {model_id}")
            
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        info = MODEL_REGISTRY[model_id]
        model_cls = info["class"]
        model = model_cls().to(self.device)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        loaded = False
        for wp in info["weight_paths"]:
            full_path = os.path.join(project_root, wp)
            if os.path.exists(full_path):
                try:
                    state_dict = torch.load(full_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"[ModelManager] Successfully loaded pre-trained weights for '{model_id}' from: {full_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[ModelManager] Warning: Failed to load weights from {full_path}: {e}")

        if not loaded:
            print(f"[ModelManager] Warning: Pre-trained weights not found for '{model_id}'. Running with initialized parameters.")

        model.eval()
        self.loaded_models[model_id] = model
        return model

    @torch.no_grad()
    def restore_image(self, model_id: str, degraded_pil: Image.Image) -> Image.Image:
        """
        Runs restoration inference using requested model on PIL Image input.
        """
        model = self.get_model(model_id)
        
        orig_w, orig_h = degraded_pil.size
        
        # Preprocess PIL image to Tensor [1, 3, 128, 128]
        transform = T.Compose([
            T.Resize((128, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])
        
        tensor_in = transform(degraded_pil.convert('RGB')).unsqueeze(0).to(self.device)
        
        # Forward pass
        tensor_out = model(tensor_in)
            
        # Postprocess back to PIL Image resized to original dimensions
        tensor_out = torch.clamp(tensor_out.squeeze(0).cpu(), 0.0, 1.0)
        to_pil = T.ToPILImage()
        restored_pil = to_pil(tensor_out).resize((orig_w, orig_h), Image.Resampling.BICUBIC)
        
        return restored_pil
