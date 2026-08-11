import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRestoration2D(nn.Module):
    """
    Model 1: Deep 2D Convolutional Neural Network (2D CNN).
    Uses sequential 2D convolutions with residual learning to perform image restoration
    at full 2D spatial resolution without encoder-decoder downsampling.
    Input: [B, 3, H, W] -> Output: [B, 3, H, W]
    """
    def __init__(self, in_channels=3, out_channels=3, num_layers=8, num_features=64):
        super(CNNRestoration2D, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features)
            ))
        self.body = nn.ModuleList(layers)
        
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.head(x)
        res = feat
        for block in self.body:
            res = res + block(res)
        out = self.tail(res)
        return out


class ConvolutionalAutoencoder(nn.Module):
    """
    Model 2: Convolutional Autoencoder.
    Encodes degraded image into a compressed latent bottleneck representation
    and decodes back to high-resolution clean image with spatial skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3, base_features=64):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Encoder Blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        # Latent Bottleneck Block
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 4, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Blocks with Transposed Convolutions
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2) # 32 -> 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2) # 64 -> 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(base_features, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out

# Alias for backwards compatibility
CNNAutoencoder = CNNRestoration2D
