import torch
import argparse
import yaml
import os
from torchvision.utils import save_image

from models import HierarchicalGenerator, HybridGenerator, DiffusionModel


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_hierarchical(model_path, output_dir, num_samples, size, device, config):
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            global_noise = torch.randn(1, config['model']['hierarchical']['latent_dim'], device=device)
            local_noise = torch.randn(1, 3, size, size, device=device)
            
            output, _ = generator(global_noise, local_noise)
            output = (output + 1) / 2  # Denormalize
            
            save_image(output, os.path.join(output_dir, f'hierarchical_sample_{i+1}.png'))


def generate_hybrid(model_path, output_dir, num_samples, size, device, config):
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            noise = torch.randn(1, config['model']['hybrid']['latent_dim'], size, size, device=device)
            output = generator(noise)
            output = (output + 1) / 2  # Denormalize
            
            save_image(output, os.path.join(output_dir, f'hybrid_sample_{i+1}.png'))


def generate_diffusion(model_path, output_dir, num_samples, size, device, config):
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            output = model.p_sample_loop((1, 3, size, size), device)
            output = (output + 1) / 2  # Denormalize
            
            save_image(output, os.path.join(output_dir, f'diffusion_sample_{i+1}.png'))


def main():
    parser = argparse.ArgumentParser(description='Generate texture samples')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['hierarchical', 'hybrid', 'diffusion'],
                       help='Type of model')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--size', type=int, default=256, help='Output image size')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print(f"Generating {args.num_samples} samples using {args.model_type} model...")
    
    if args.model_type == 'hierarchical':
        generate_hierarchical(args.model_path, args.output_dir, args.num_samples, 
                            args.size, device, config)
    elif args.model_type == 'hybrid':
        generate_hybrid(args.model_path, args.output_dir, args.num_samples, 
                       args.size, device, config)
    elif args.model_type == 'diffusion':
        generate_diffusion(args.model_path, args.output_dir, args.num_samples, 
                          args.size, device, config)
    
    print(f"Samples saved to {args.output_dir}")


if __name__ == '__main__':
    main()
