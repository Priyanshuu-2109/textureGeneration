import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
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


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    """
    U-Net backbone for diffusion model with large receptive field.
    Designed to capture patch-level statistics effectively.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Encoder
        self.down1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        
        # Decoder
        self.up1 = ResidualBlock(base_channels * 8 + base_channels * 4, base_channels * 4, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.up3 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Encoder
        x1 = self.down1(x, t)
        x2 = self.down2(F.avg_pool2d(x1, 2), t)
        x3 = self.down3(F.avg_pool2d(x2, 2), t)
        
        # Bottleneck
        x4 = self.bottleneck(F.avg_pool2d(x3, 2), t)
        
        # Decoder with skip connections
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, x3], dim=1), t)
        
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, x2], dim=1), t)
        
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, x1], dim=1), t)
        
        return self.output(x)


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Model for texture synthesis.
    Trains on patches from multiple images to learn texture distribution.
    """
    def __init__(self, image_size=256, timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 unet_channels=64, unet_depth=3):
        super().__init__()
        self.image_size = image_size
        self.timesteps = timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # U-Net model
        self.model = UNet(in_channels=3, out_channels=3, base_channels=unet_channels)
    
    def register_schedule(self, device):
        """Register noise schedule to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, t):
        """Sample from p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, shape, device):
        """Generate samples by iteratively denoising."""
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        
        return img
    
    def forward(self, x_start, t=None):
        """Training forward pass."""
        if t is None:
            t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device).long()
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        return predicted_noise, noise
