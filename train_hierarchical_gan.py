import torch
import torch.nn as nn
import torch.optim as optim
from utils.tensorboard_safe import SummaryWriter
import yaml
import os
from tqdm import tqdm
import argparse

from models import HierarchicalGenerator, GlobalDiscriminator, PatchDiscriminator
from utils import get_dataloader, StyleLoss, FFTLoss, PerceptualLoss, LSGANLoss


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(generator, global_disc, patch_disc, dataloader, 
                optim_g, optim_global_d, optim_patch_d,
                style_loss, fft_loss, perceptual_loss, gan_loss,
                device, epoch, config, writer):
    """Train one epoch."""
    generator.train()
    global_disc.train()
    patch_disc.train()
    
    total_g_loss = 0
    total_global_d_loss = 0
    total_patch_d_loss = 0
    
    for batch_idx, (real_images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Prepare noise
        global_noise = torch.randn(batch_size, config['model']['hierarchical']['latent_dim'], device=device)
        local_noise = torch.randn_like(real_images)
        
        # #region agent log
        import json
        log_path = '/Users/priyanshusrivastav/Desktop/textureGeneration/.cursor/debug.log'
        try:
            with open(log_path, 'a') as f:
                log_entry = {
                    "location": "train_hierarchical_gan.py:39",
                    "message": "Noise preparation",
                    "data": {
                        "real_images_shape": list(real_images.shape),
                        "local_noise_shape": list(local_noise.shape),
                        "current_size": real_images.shape[2]
                    },
                    "timestamp": int(__import__('time').time() * 1000),
                    "runId": "debug_run",
                    "hypothesisId": "C"
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        # ========== Train Discriminators ==========
        # Global Discriminator
        optim_global_d.zero_grad()
        
        # Downsample real images for global discriminator
        real_global = nn.functional.interpolate(real_images, size=(64, 64), mode='bilinear', align_corners=False)
        
        fake_output, global_output = generator(global_noise, local_noise)
        fake_global = global_output.detach()
        
        real_global_pred = global_disc(real_global)
        fake_global_pred = global_disc(fake_global)
        
        loss_global_d = (gan_loss(real_global_pred, True) + gan_loss(fake_global_pred, False)) / 2
        loss_global_d.backward()
        optim_global_d.step()
        
        # Patch Discriminator
        optim_patch_d.zero_grad()
        
        # Extract random patches from real images
        patch_size = config['model']['hierarchical']['patchgan_patch_size']
        h, w = real_images.shape[2:]
        i = torch.randint(0, h - patch_size + 1, (1,)).item()
        j = torch.randint(0, w - patch_size + 1, (1,)).item()
        real_patch = real_images[:, :, i:i+patch_size, j:j+patch_size]
        
        fake_patch = fake_output.detach()[:, :, i:i+patch_size, j:j+patch_size]
        
        real_patch_pred = patch_disc(real_patch)
        fake_patch_pred = patch_disc(fake_patch)
        
        loss_patch_d = (gan_loss(real_patch_pred, True) + gan_loss(fake_patch_pred, False)) / 2
        loss_patch_d.backward()
        optim_patch_d.step()
        
        # ========== Train Generator ==========
        optim_g.zero_grad()
        
        fake_output, global_output = generator(global_noise, local_noise)
        fake_global = global_output
        
        # Adversarial losses
        fake_global_pred = global_disc(fake_global)
        fake_patch_pred = patch_disc(fake_output)
        
        loss_adv_global = gan_loss(fake_global_pred, True)
        loss_adv_patch = gan_loss(fake_patch_pred, True)
        loss_adv = (loss_adv_global + loss_adv_patch) / 2
        
        # Auxiliary losses
        loss_style = style_loss(fake_output, real_images)
        loss_fft = fft_loss(fake_output, real_images)
        loss_perceptual = perceptual_loss(fake_output, real_images)
        
        # Total generator loss
        loss_g = (config['losses']['adversarial_weight'] * loss_adv +
                 config['losses']['style_weight'] * loss_style +
                 config['losses']['fft_weight'] * loss_fft +
                 config['losses']['perceptual_weight'] * loss_perceptual)
        
        loss_g.backward()
        optim_g.step()
        
        # Accumulate losses
        total_g_loss += loss_g.item()
        total_global_d_loss += loss_global_d.item()
        total_patch_d_loss += loss_patch_d.item()
        
        # Logging
        if batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/Generator', loss_g.item(), global_step)
            writer.add_scalar('Loss/Global_Discriminator', loss_global_d.item(), global_step)
            writer.add_scalar('Loss/Patch_Discriminator', loss_patch_d.item(), global_step)
            writer.add_scalar('Loss/Adversarial', loss_adv.item(), global_step)
            writer.add_scalar('Loss/Style', loss_style.item(), global_step)
            writer.add_scalar('Loss/FFT', loss_fft.item(), global_step)
            writer.add_scalar('Loss/Perceptual', loss_perceptual.item(), global_step)
    
    return (total_g_loss / len(dataloader),
            total_global_d_loss / len(dataloader),
            total_patch_d_loss / len(dataloader))


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
    generator = HierarchicalGenerator(
        latent_dim=config['model']['hierarchical']['latent_dim'],
        global_channels=config['model']['hierarchical']['global_channels'],
        local_channels=config['model']['hierarchical']['local_channels'],
        final_size=config['data']['image_size']
    ).to(device)
    
    global_disc = GlobalDiscriminator(
        base_channels=config['model']['hierarchical']['global_channels']
    ).to(device)
    
    patch_disc = PatchDiscriminator(
        base_channels=config['model']['hierarchical']['global_channels'],
        patch_size=config['model']['hierarchical']['patchgan_patch_size']
    ).to(device)
    
    # Optimizers
    optim_g = optim.Adam(
        generator.parameters(),
        lr=config['training']['learning_rate_g'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    optim_global_d = optim.Adam(
        global_disc.parameters(),
        lr=config['training']['learning_rate_d'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    optim_patch_d = optim.Adam(
        patch_disc.parameters(),
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
        global_disc.load_state_dict(checkpoint['global_disc'])
        patch_disc.load_state_dict(checkpoint['patch_disc'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_global_d.load_state_dict(checkpoint['optim_global_d'])
        optim_patch_d.load_state_dict(checkpoint['optim_patch_d'])
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
        g_loss, global_d_loss, patch_d_loss = train_epoch(
            generator, global_disc, patch_disc, dataloader,
            optim_g, optim_global_d, optim_patch_d,
            style_loss, fft_loss, perceptual_loss, gan_loss,
            device, epoch, config, writer
        )
        
        print(f"Epoch {epoch}: G_loss={g_loss:.4f}, Global_D_loss={global_d_loss:.4f}, Patch_D_loss={patch_d_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'global_disc': global_disc.state_dict(),
                'patch_disc': patch_disc.state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_global_d': optim_global_d.state_dict(),
                'optim_patch_d': optim_patch_d.state_dict(),
            }
            torch.save(checkpoint, 
                      os.path.join(config['output']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Generate samples
            generator.eval()
            with torch.no_grad():
                sample_noise_global = torch.randn(4, config['model']['hierarchical']['latent_dim'], device=device)
                sample_noise_local = torch.randn(4, 3, current_size, current_size, device=device)
                samples, _ = generator(sample_noise_global, sample_noise_local)
                samples = (samples + 1) / 2  # Denormalize
                
                from torchvision.utils import save_image
                save_image(samples, 
                          os.path.join(config['output']['sample_dir'], f'samples_epoch_{epoch+1}.png'),
                          nrow=2)
    
    writer.close()


if __name__ == '__main__':
    main()
