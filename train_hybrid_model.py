import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
import argparse

from models import HybridGenerator, HybridDiscriminator
from utils import get_dataloader, StyleLoss, FFTLoss, PerceptualLoss, LSGANLoss


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(generator, discriminator, dataloader, optim_g, optim_d,
                style_loss, fft_loss, perceptual_loss, gan_loss,
                device, epoch, config, writer):
    """Train one epoch."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    for batch_idx, (real_images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Prepare noise
        noise = torch.randn(batch_size, config['model']['hybrid']['latent_dim'], 
                           config['data']['image_size'], config['data']['image_size'], 
                           device=device)
        
        # ========== Train Discriminator ==========
        optim_d.zero_grad()
        
        fake_images = generator(noise).detach()
        
        real_pred = discriminator(real_images)
        fake_pred = discriminator(fake_images)
        
        loss_d = (gan_loss(real_pred, True) + gan_loss(fake_pred, False)) / 2
        loss_d.backward()
        optim_d.step()
        
        # ========== Train Generator ==========
        optim_g.zero_grad()
        
        fake_images = generator(noise)
        fake_pred = discriminator(fake_images)
        
        # Adversarial loss
        loss_adv = gan_loss(fake_pred, True)
        
        # Auxiliary losses
        loss_style = style_loss(fake_images, real_images)
        loss_fft = fft_loss(fake_images, real_images)
        loss_perceptual = perceptual_loss(fake_images, real_images)
        
        # Total generator loss
        loss_g = (config['losses']['adversarial_weight'] * loss_adv +
                 config['losses']['style_weight'] * loss_style +
                 config['losses']['fft_weight'] * loss_fft +
                 config['losses']['perceptual_weight'] * loss_perceptual)
        
        loss_g.backward()
        optim_g.step()
        
        # Accumulate losses
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        
        # Logging
        if batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/Generator', loss_g.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_d.item(), global_step)
            writer.add_scalar('Loss/Adversarial', loss_adv.item(), global_step)
            writer.add_scalar('Loss/Style', loss_style.item(), global_step)
            writer.add_scalar('Loss/FFT', loss_fft.item(), global_step)
            writer.add_scalar('Loss/Perceptual', loss_perceptual.item(), global_step)
    
    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Override data directory from config')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['sample_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    
    # Models
    generator = HybridGenerator(
        latent_dim=config['model']['hybrid']['latent_dim'],
        base_channels=config['model']['hybrid']['base_channels'],
        transformer_dim=config['model']['hybrid']['transformer_dim'],
        num_heads=config['model']['hybrid']['num_heads'],
        num_layers=config['model']['hybrid']['num_layers'],
        image_size=config['data']['image_size']
    ).to(device)
    
    discriminator = HybridDiscriminator(
        base_channels=config['model']['hybrid']['base_channels'],
        use_transformer=False
    ).to(device)
    
    # Optimizers
    optim_g = optim.Adam(
        generator.parameters(),
        lr=config['training']['learning_rate_g'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    optim_d = optim.Adam(
        discriminator.parameters(),
        lr=config['training']['learning_rate_d'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    # Losses
    style_loss = StyleLoss().to(device)
    fft_loss = FFTLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    gan_loss = LSGANLoss()
    
    # Data loader
    current_size = config['training']['start_size'] if config['training']['progressive_resizing'] else config['data']['image_size']
    dataloader = get_dataloader(
        data_dir=config['data']['data_dir'],
        image_size=current_size,
        patch_size=config['data']['patch_size'],
        batch_size=config['training']['batch_size'],
        augment=True
    )
    
    # TensorBoard
    writer = SummaryWriter(config['output']['log_dir'])
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Progressive resizing
        if config['training']['progressive_resizing']:
            if epoch in config['training']['resize_epochs']:
                idx = config['training']['resize_epochs'].index(epoch)
                sizes = [config['training']['start_size'], 
                         config['training']['start_size'] * 2,
                         config['data']['image_size']]
                if idx < len(sizes):
                    current_size = sizes[idx]
                    print(f"Resizing to {current_size}x{current_size}")
                    dataloader = get_dataloader(
                        data_dir=config['data']['data_dir'],
                        image_size=current_size,
                        patch_size=config['data']['patch_size'],
                        batch_size=config['training']['batch_size'],
                        augment=True
                    )
        
        # Train
        g_loss, d_loss = train_epoch(
            generator, discriminator, dataloader,
            optim_g, optim_d,
            style_loss, fft_loss, perceptual_loss, gan_loss,
            device, epoch, config, writer
        )
        
        print(f"Epoch {epoch}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
            }
            torch.save(checkpoint, 
                      os.path.join(config['output']['checkpoint_dir'], f'hybrid_checkpoint_epoch_{epoch+1}.pth'))
            
            # Generate samples
            generator.eval()
            with torch.no_grad():
                sample_noise = torch.randn(4, config['model']['hybrid']['latent_dim'], 
                                         current_size, current_size, device=device)
                samples = generator(sample_noise)
                samples = (samples + 1) / 2  # Denormalize
                
                from torchvision.utils import save_image
                save_image(samples, 
                          os.path.join(config['output']['sample_dir'], f'hybrid_samples_epoch_{epoch+1}.png'),
                          nrow=2)
    
    writer.close()


if __name__ == '__main__':
    main()
