import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, time_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        # Add time embedding
        time_emb = self.relu(self.time_mlp(time_emb))
        h = h + time_emb[(..., ) + (None, ) * 2]
        h = self.relu(self.bn2(self.conv2(h)))
        return h

class DiffusionUNet(nn.Module):
    """
    UNet Backbone for DDPM noise estimation epsilon_theta(x_t, t).
    """
    def __init__(self, in_channels=3, out_channels=3, base_dim=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )
        
        # Down
        self.b1 = ConvBlock(in_channels, base_dim, time_emb_dim)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        
        self.b2 = ConvBlock(base_dim * 2, base_dim * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, stride=2, padding=1)
        
        # Bottleneck
        self.mid = ConvBlock(base_dim * 4, base_dim * 4, time_emb_dim)
        
        # Up
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.b_up2 = ConvBlock(base_dim * 4, base_dim * 2, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.b_up1 = ConvBlock(base_dim * 2, base_dim, time_emb_dim)
        
        self.out_conv = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x, timesteps):
        t_emb = self.time_mlp(timesteps)
        
        h1 = self.b1(x, t_emb)
        d1 = self.down1(h1)
        
        h2 = self.b2(d1, t_emb)
        d2 = self.down2(h2)
        
        m = self.mid(d2, t_emb)
        
        u2 = self.up2(m)
        u2 = torch.cat([u2, h2], dim=1)
        h_up2 = self.b_up2(u2, t_emb)
        
        u1 = self.up1(h_up2)
        u1 = torch.cat([u1, h1], dim=1)
        h_up1 = self.b_up1(u1, t_emb)
        
        return self.out_conv(h_up1)

class DDPM(nn.Module):
    """
    Model 5: Denoising Diffusion Probabilistic Model (DDPM).
    Handles forward noise process, reverse sampling, and image restoration.
    """
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, in_channels=3):
        super().__init__()
        self.timesteps = timesteps
        self.model = DiffusionUNet(in_channels=in_channels, out_channels=in_channels)
        
        # Define beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x, t):
        # Predict noise added at timestep t
        return self.model(x, t)

    @torch.no_grad()
    def restore(self, degraded_img, num_steps=20):
        """
        Fast reverse diffusion process for restoration starting from degraded_img.
        """
        self.eval()
        device = degraded_img.device
        b = degraded_img.shape[0]
        
        # Start at a low noise timestep proportional to fast restoration (e.g. t=100)
        step_indices = torch.linspace(self.timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        x = degraded_img.clone()
        for i, t_val in enumerate(step_indices):
            t_tensor = torch.full((b,), t_val, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t_tensor)
            
            alpha_t = self.alphas[t_val]
            alpha_cumprod_t = self.alphas_cumprod[t_val]
            beta_t = self.betas[t_val]
            
            if t_val > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            x = torch.clamp(x, 0.0, 1.0)
            
        return x
