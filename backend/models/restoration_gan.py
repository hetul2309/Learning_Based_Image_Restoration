import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models

class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return x3 * 0.2 + x

class RestorationGANGenerator(nn.Module):
    """
    Model 3: Restoration GAN Generator with Dense Residual Blocks.
    Input: Degraded Image [B, 3, H, W] -> Output: Restored Image [B, 3, H, W]
    """
    def __init__(self, in_channels=3, out_channels=3, num_blocks=4, base_channels=64):
        super(RestorationGANGenerator, self).__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.body = nn.Sequential(*[DenseBlock(base_channels) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.head(x)
        res = self.conv_body(self.body(feat))
        out = self.tail(feat + res)
        return out

class PatchDiscriminator(nn.Module):
    """
    Markovian PatchGAN Discriminator for fine visual detail discrimination.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(PatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class VGGLoss(nn.Module):
    """
    Perceptual VGG Feature Loss calculated using pretrained VGG-19 features.
    """
    def __init__(self):
        super(VGGLoss, self).__init__()
        try:
            vgg = torchvision_models.vgg19(weights=torchvision_models.VGG19_Weights.DEFAULT)
        except Exception:
            vgg = torchvision_models.vgg19(pretrained=True)
            
        self.feature_extractor = nn.Sequential(*list(vgg.features[:16])).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        # Normalize to VGG standard expectation if range [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=sr.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=sr.device).view(1, 3, 1, 1)
        
        sr_norm = (sr - mean) / std
        hr_norm = (hr - mean) / std
        
        sr_feat = self.feature_extractor(sr_norm)
        hr_feat = self.feature_extractor(hr_norm)
        
        return F.l1_loss(sr_feat, hr_feat)
