import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class StyleLoss(nn.Module):
    """
    Gram matrix style loss for matching texture statistics.
    Computes Gram matrices on VGG features and compares them.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def gram_matrix(self, features):
        """Compute Gram matrix for style loss."""
        batch_size, channels, height, width = features.size()
        features_flat = features.view(batch_size, channels, height * width)
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
        return gram / (channels * height * width)
    
    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        
        gen_gram = self.gram_matrix(gen_features)
        target_gram = self.gram_matrix(target_features)
        
        return F.mse_loss(gen_gram, target_gram)


class FFTLoss(nn.Module):
    """
    FFT-based loss to match low-frequency content and power spectrum.
    Encourages global structure consistency.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, generated, target):
        # Convert to frequency domain
        gen_fft = torch.fft.fft2(generated, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # Compute power spectrum
        gen_power = torch.abs(gen_fft) ** 2
        target_power = torch.abs(target_fft) ** 2
        
        # Low-frequency emphasis (weight by inverse frequency)
        h, w = generated.shape[-2:]
        y = torch.arange(h, device=generated.device).float()
        x = torch.arange(w, device=generated.device).float()
        Y, X = torch.meshgrid(y, x, indexing='ij')
        freq_weight = 1.0 / (1.0 + torch.sqrt((Y - h/2)**2 + (X - w/2)**2))
        freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)
        
        # Weighted MSE on power spectrum
        loss = F.mse_loss(gen_power * freq_weight, target_power * freq_weight)
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features for high-level similarity.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        
        return F.mse_loss(gen_features, target_features)


class LSGANLoss(nn.Module):
    """Least Squares GAN loss for stable training."""
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
    
    def __call__(self, predictions, is_real):
        if is_real:
            labels = torch.ones_like(predictions) * self.real_label
        else:
            labels = torch.ones_like(predictions) * self.fake_label
        
        return F.mse_loss(predictions, labels)


class HingeGANLoss(nn.Module):
    """Hinge loss for GAN training."""
    def __init__(self):
        super().__init__()
    
    def __call__(self, predictions, is_real):
        if is_real:
            return F.relu(1 - predictions).mean()
        else:
            return F.relu(1 + predictions).mean()
