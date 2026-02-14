import torch
import argparse
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models import HierarchicalGenerator, HybridGenerator, DiffusionModel
from utils import compute_sifid, compute_lpips, compute_seam_continuity


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_real_images(data_dir, num_images, image_size, device):
    """Load real images from directory."""
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
             if f.lower().endswith(ext.replace('*', ''))]
        )
    image_paths = sorted(image_paths)[:num_images]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0)
        images.append(img)
    
    return torch.cat(images, dim=0).to(device)


def generate_samples_hierarchical(model_path, num_samples, size, device, config):
    """Generate samples using Hierarchical GAN."""
    generator = HierarchicalGenerator(
        latent_dim=config['model']['hierarchical']['latent_dim'],
        global_channels=config['model']['hierarchical']['global_channels'],
        local_channels=config['model']['hierarchical']['local_channels'],
        final_size=size
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            global_noise = torch.randn(1, config['model']['hierarchical']['latent_dim'], device=device)
            local_noise = torch.randn(1, 3, size, size, device=device)
            output, _ = generator(global_noise, local_noise)
            samples.append(output)
    
    return torch.cat(samples, dim=0)


def generate_samples_hybrid(model_path, num_samples, size, device, config):
    """Generate samples using Hybrid CNN-Transformer."""
    generator = HybridGenerator(
        latent_dim=config['model']['hybrid']['latent_dim'],
        base_channels=config['model']['hybrid']['base_channels'],
        transformer_dim=config['model']['hybrid']['transformer_dim'],
        num_heads=config['model']['hybrid']['num_heads'],
        num_layers=config['model']['hybrid']['num_layers'],
        image_size=size
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            noise = torch.randn(1, config['model']['hybrid']['latent_dim'], size, size, device=device)
            output = generator(noise)
            samples.append(output)
    
    return torch.cat(samples, dim=0)


def generate_samples_diffusion(model_path, num_samples, size, device, config):
    """Generate samples using Diffusion Model."""
    model = DiffusionModel(
        image_size=size,
        timesteps=config['model']['diffusion']['timesteps'],
        beta_start=config['model']['diffusion']['beta_start'],
        beta_end=config['model']['diffusion']['beta_end'],
        unet_channels=config['model']['diffusion']['unet_channels']
    ).to(device)
    
    model.register_schedule(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            output = model.p_sample_loop((1, 3, size, size), device)
            samples.append(output)
    
    return torch.cat(samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description='Evaluate texture synthesis models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['hierarchical', 'hybrid', 'diffusion'],
                       help='Type of model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with real texture images')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print(f"Evaluating {args.model_type} model...")
    print(f"Generating {args.num_samples} samples...")
    
    # Generate samples
    if args.model_type == 'hierarchical':
        generated_images = generate_samples_hierarchical(
            args.model_path, args.num_samples, args.size, device, config
        )
    elif args.model_type == 'hybrid':
        generated_images = generate_samples_hybrid(
            args.model_path, args.num_samples, args.size, device, config
        )
    elif args.model_type == 'diffusion':
        generated_images = generate_samples_diffusion(
            args.model_path, args.num_samples, args.size, device, config
        )
    
    # Load real images
    print("Loading real images...")
    real_images = load_real_images(args.data_dir, min(args.num_samples, 40), args.size, device)
    
    # Denormalize for metrics
    generated_images = (generated_images + 1) / 2
    real_images_denorm = (real_images + 1) / 2
    
    # Compute metrics
    results = {}
    
    if config['evaluation']['compute_sifid']:
        print("Computing SIFID...")
        sifid = compute_sifid(generated_images, real_images, device)
        results['SIFID'] = sifid
        print(f"SIFID: {sifid:.4f}")
    
    if config['evaluation']['compute_lpips']:
        print("Computing LPIPS...")
        lpips_score = compute_lpips(generated_images, real_images, device)
        results['LPIPS'] = lpips_score
        print(f"LPIPS: {lpips_score:.4f}")
    
    if config['evaluation']['compute_seam_continuity']:
        print("Computing seam continuity...")
        seam_scores = []
        for img in generated_images:
            score = compute_seam_continuity(img.cpu().numpy())
            seam_scores.append(score)
        avg_seam = np.mean(seam_scores)
        results['Seam Continuity'] = avg_seam
        print(f"Average Seam Continuity: {avg_seam:.4f}")
    
    # Print summary
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    results_file = f'evaluation_results_{args.model_type}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Checkpoint: {args.model_path}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Image size: {args.size}\n\n")
        f.write("Metrics:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
