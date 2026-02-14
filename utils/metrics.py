import torch
import numpy as np
from scipy import linalg
import lpips
from torchvision.models import inception_v3
from torchvision.transforms import functional as F


def compute_sifid(generated_images, real_images, device='cuda'):
    """
    Single-Image FID: FID computed on single images.
    Measures perceptual quality and diversity.
    """
    # Load Inception model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.fc = torch.nn.Identity()
    
    def get_features(images):
        features = []
        with torch.no_grad():
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = F.resize(img, (299, 299))
                feat = inception(img)
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)
    
    gen_features = get_features(generated_images)
    real_features = get_features(real_images)
    
    # Compute statistics
    mu1, sigma1 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    mu2, sigma2 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    
    # Compute FID
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid


def compute_lpips(generated_images, real_images, device='cuda'):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between images.
    """
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Normalize images to [-1, 1] if needed
    gen_tensor = generated_images.clone()
    real_tensor = real_images.clone()
    
    if gen_tensor.max() > 1.0:
        gen_tensor = gen_tensor / 255.0
    if real_tensor.max() > 1.0:
        real_tensor = real_tensor / 255.0
    
    gen_tensor = gen_tensor * 2.0 - 1.0
    real_tensor = real_tensor * 2.0 - 1.0
    
    distances = []
    with torch.no_grad():
        for gen_img, real_img in zip(gen_tensor, real_tensor):
            if gen_img.dim() == 3:
                gen_img = gen_tensor.unsqueeze(0)
            if real_img.dim() == 3:
                real_img = real_tensor.unsqueeze(0)
            
            dist = loss_fn(gen_img.to(device), real_img.to(device))
            distances.append(dist.item())
    
    return np.mean(distances)


def compute_seam_continuity(image, patch_size=64, overlap=16):
    """
    Compute seam continuity metric by checking border differences.
    Lower values indicate better continuity.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = np.transpose(image, (1, 2, 0))
    
    h, w = image.shape[:2]
    
    # Extract patches with overlap
    stride = patch_size - overlap
    border_errors = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            
            # Check right border
            if j + patch_size < w:
                right_border = patch[:, -overlap:]
                next_patch_left = image[i:i+patch_size, j+stride:j+stride+overlap]
                error = np.mean(np.abs(right_border - next_patch_left))
                border_errors.append(error)
            
            # Check bottom border
            if i + patch_size < h:
                bottom_border = patch[-overlap:, :]
                next_patch_top = image[i+stride:i+stride+overlap, j:j+patch_size]
                error = np.mean(np.abs(bottom_border - next_patch_top))
                border_errors.append(error)
    
    return np.mean(border_errors) if border_errors else 0.0
