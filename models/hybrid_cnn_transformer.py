import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for capturing long-range dependencies."""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CNNBlock(nn.Module):
    """Convolutional block for local pattern extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class HybridGenerator(nn.Module):
    """
    Hybrid CNN-Transformer Generator: Combines convolutional layers for local
    patterns with transformer layers for global texture correlations.
    """
    def __init__(self, latent_dim=128, base_channels=64, transformer_dim=256,
                 num_heads=8, num_layers=4, image_size=256, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Initial CNN layers for local feature extraction
        self.cnn_encoder = nn.Sequential(
            CNNBlock(latent_dim, base_channels),
            CNNBlock(base_channels, base_channels * 2),
            CNNBlock(base_channels * 2, base_channels * 4),
        )
        
        # Project to transformer dimension
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * base_channels * 4, transformer_dim)
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, transformer_dim)
        )
        
        # Transformer blocks for global correlation
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, num_heads, transformer_dim // num_heads)
            for _ in range(num_layers)
        ])
        
        # Project back to feature maps
        self.patch_unembed = nn.Sequential(
            nn.Linear(transformer_dim, patch_size * patch_size * base_channels * 4),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=image_size // patch_size, w=image_size // patch_size,
                     p1=patch_size, p2=patch_size, c=base_channels * 4)
        )
        
        # CNN decoder for refinement
        self.cnn_decoder = nn.Sequential(
            CNNBlock(base_channels * 4, base_channels * 2),
            CNNBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, noise):
        # CNN encoding for local patterns
        x = self.cnn_encoder(noise)
        
        # Transform to patches and embed
        x = self.patch_embed(x)
        x = x + self.pos_embedding
        
        # Transformer for global correlations
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Transform back to feature maps
        x = self.patch_unembed(x)
        
        # CNN decoding for final output
        output = self.cnn_decoder(x)
        
        return output


class HybridDiscriminator(nn.Module):
    """
    Discriminator for hybrid model: Uses CNN with optional transformer layers.
    """
    def __init__(self, base_channels=64, use_transformer=False, transformer_dim=256):
        super().__init__()
        self.use_transformer = use_transformer
        
        # CNN feature extraction
        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
        )
        
        if use_transformer:
            # Optional transformer for global discrimination
            h, w = 32, 32  # After 3 downsampling layers
            self.patch_embed = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(base_channels * 4, transformer_dim)
            )
            self.transformer = TransformerBlock(transformer_dim)
            self.patch_unembed = nn.Linear(transformer_dim, base_channels * 4)
            self.final_conv = nn.Sequential(
                Rearrange('b (h w) c -> b c h w', h=h, w=w),
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
            )
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
            )
    
    def forward(self, x):
        x = self.cnn_features(x)
        
        if self.use_transformer:
            x = self.patch_embed(x)
            x = self.transformer(x)
            x = self.patch_unembed(x)
            x = self.final_conv(x)
        else:
            x = self.final_conv(x)
        
        return x.view(x.size(0), -1).mean(1)
