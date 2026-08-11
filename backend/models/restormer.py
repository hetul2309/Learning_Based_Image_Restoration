import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Self-Attention (MDTA) module.
    Applies self-attention across channel dimension instead of spatial dimension.
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN) module.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class Restormer(nn.Module):
    """
    Model 4: Transformer-based Image Restoration Architecture (Restormer/SwinIR).
    Input: [B, 3, H, W] -> Output: [B, 3, H, W]
    """
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_blocks=[2, 2, 2, 2], heads=[1, 2, 4, 8]):
        super(Restormer, self).__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0]) for _ in range(num_blocks[0])])
        
        # Downsample 1
        self.down1_2 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        
        # Encoder Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim * 2, num_heads=heads[1]) for _ in range(num_blocks[1])])

        # Bottleneck
        self.bottleneck = nn.Sequential(*[TransformerBlock(dim=dim * 2, num_heads=heads[2]) for _ in range(num_blocks[2])])

        # Upsample 1
        self.up2_1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1)
        self.reduce_chan = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0]) for _ in range(num_blocks[3])])

        self.output = nn.Sequential(
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, inp_img):
        feat = self.patch_embed(inp_img)
        
        e1 = self.encoder_level1(feat)
        d1 = self.down1_2(e1)
        
        e2 = self.encoder_level2(d1)
        b = self.bottleneck(e2)
        
        u1 = self.up2_1(b)
        u1 = self.reduce_chan(torch.cat([u1, e1], dim=1))
        d_out = self.decoder_level1(u1)
        
        out = self.output(d_out + feat)
        return out
