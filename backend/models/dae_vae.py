import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingVAE(nn.Module):
    """
    Model 2: Denoising Variational Autoencoder (VAE).
    Encodes degraded images to mean and logvar latent representation, reparameterizes,
    and decodes back to clean image space.
    """
    def __init__(self, in_channels=3, latent_dim=256, base_channels=32):
        super(DenoisingVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),  # H/2
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), # H/4
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1), # H/8
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1), # H/16
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_mu = nn.Linear(base_channels * 8 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8 * 8 * 8, latent_dim)
        
        # Decoder Projection
        self.decoder_input = nn.Linear(latent_dim, base_channels * 8 * 8 * 8)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1), # H/8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1), # H/4
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1), # H/2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1), # H
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 8, 8)
        out = self.decoder(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon

    def loss_function(self, recon_x, x, mu, logvar, kl_weight=0.0001):
        recon_loss = F.l1_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
