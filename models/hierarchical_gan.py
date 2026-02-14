import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalModule(nn.Module):
    """
    Global Module: Generates coarse structure at low resolution.
    Uses U-Net-like architecture to capture large-scale patterns.
    """
    def __init__(self, latent_dim=128, base_channels=64, output_size=64):
        super().__init__()
        self.output_size = output_size
        self.latent_dim = latent_dim
        
        # Project latent to spatial dimensions
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 4 * 4 * 4),
            nn.ReLU()
        )
        
        # Initial convolution from projected latent
        self.init_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 1, 0),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU()
        )
        
        # Decoder blocks
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, noise):
        # noise shape: (batch, latent_dim)
        b = noise.size(0)
        
        # Project latent to spatial feature map
        x = self.latent_proj(noise)
        x = x.view(b, -1, 4, 4)
        
        # Decode to output size
        x = self.init_conv(x)
        x = self.dec1(x)
        x = self.dec2(x)
        output = self.dec3(x)
        
        return output


class LocalRefinementModule(nn.Module):
    """
    Local Refinement Module: Adds fine texture detail at high resolution.
    Takes upsampled global output and adds local details.
    """
    def __init__(self, base_channels=128):
        super().__init__()
        
        # Input: upsampled global (3 channels) + noise (3 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, global_upsampled, local_noise):
        # #region agent log
        import json
        log_path = '/Users/priyanshusrivastav/Desktop/textureGeneration/.cursor/debug.log'
        try:
            with open(log_path, 'a') as f:
                log_entry = {
                    "location": "hierarchical_gan.py:100",
                    "message": "LocalRefinementModule.forward entry",
                    "data": {
                        "global_upsampled_shape": list(global_upsampled.shape),
                        "local_noise_shape": list(local_noise.shape),
                        "spatial_dims_match": list(global_upsampled.shape[2:]) == list(local_noise.shape[2:])
                    },
                    "timestamp": int(__import__('time').time() * 1000),
                    "runId": "debug_run",
                    "hypothesisId": "B"
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        x = torch.cat([global_upsampled, local_noise], dim=1)
        x = self.conv1(x)
        x = self.conv2(x) + x  # Residual
        x = self.conv3(x) + x  # Residual
        x = self.conv4(x)
        output = self.output(x)
        
        # Residual connection to global
        return output + global_upsampled


class HierarchicalGenerator(nn.Module):
    """
    Hierarchical Generator: Combines Global and Local modules.
    """
    def __init__(self, latent_dim=128, global_channels=64, local_channels=128,
                 global_size=64, final_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_size = global_size
        self.final_size = final_size
        
        self.global_module = GlobalModule(latent_dim, global_channels, global_size)
        self.local_module = LocalRefinementModule(local_channels)
    
    def forward(self, global_noise=None, local_noise=None):
        # #region agent log
        import json
        log_path = '/Users/priyanshusrivastav/Desktop/textureGeneration/.cursor/debug.log'
        try:
            with open(log_path, 'a') as f:
                log_entry = {
                    "location": "hierarchical_gan.py:126",
                    "message": "HierarchicalGenerator.forward entry",
                    "data": {
                        "global_noise_shape": list(global_noise.shape) if global_noise is not None else None,
                        "local_noise_shape": list(local_noise.shape) if local_noise is not None else None,
                        "final_size": self.final_size
                    },
                    "timestamp": int(__import__('time').time() * 1000),
                    "runId": "debug_run",
                    "hypothesisId": "A"
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        # Generate global noise if not provided
        if global_noise is None:
            batch_size = 1 if local_noise is None else local_noise.size(0)
            global_noise = torch.randn(batch_size, self.latent_dim, 
                                      device=next(self.parameters()).device)
        
        # Generate global structure
        global_output = self.global_module(global_noise)
        
        # Upsample to final size
        global_upsampled = F.interpolate(
            global_output, 
            size=(self.final_size, self.final_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Generate local noise if not provided
        if local_noise is None:
            local_noise = torch.randn_like(global_upsampled)
        
        # #region agent log
        try:
            with open(log_path, 'a') as f:
                log_entry = {
                    "location": "hierarchical_gan.py:149",
                    "message": "Before LocalRefinementModule.forward",
                    "data": {
                        "global_upsampled_shape": list(global_upsampled.shape),
                        "local_noise_shape": list(local_noise.shape),
                        "shapes_match": list(global_upsampled.shape[2:]) == list(local_noise.shape[2:])
                    },
                    "timestamp": int(__import__('time').time() * 1000),
                    "runId": "debug_run",
                    "hypothesisId": "A"
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        # Refine with local module
        output = self.local_module(global_upsampled, local_noise)
        
        return output, global_output


class GlobalDiscriminator(nn.Module):
    """
    Global Discriminator: Enforces large-scale realism on low-res output.
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
        )
    
    def forward(self, x):
        return self.model(x).view(x.size(0), -1).mean(1)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator: Enforces local texture fidelity on random patches.
    """
    def __init__(self, base_channels=64, patch_size=70):
        super().__init__()
        self.patch_size = patch_size
        
        self.model = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
        )
    
    def forward(self, x):
        # Extract random patch if input is larger than patch_size
        if x.size(-1) > self.patch_size:
            b, c, h, w = x.shape
            i = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            j = torch.randint(0, w - self.patch_size + 1, (1,)).item()
            x = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
        
        return self.model(x)
