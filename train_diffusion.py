import torch
import torch.nn as nn
import torch.optim as optim
from utils.tensorboard_safe import SummaryWriter
import yaml
import os
from tqdm import tqdm
import argparse

from models import DiffusionModel
from utils import get_dataloader


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, device, epoch, config, writer):
    """Train one epoch."""
    model.train()
    
    total_loss = 0
    
    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images = images.to(device)
        
        # Sample random timesteps
        t = torch.randint(0, config['model']['diffusion']['timesteps'], 
                         (images.shape[0],), device=device).long()
        
        # Forward pass
        predicted_noise, noise = model(images, t)
        
        # Loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/Train', loss.item(), global_step)
    
    return total_loss / len(dataloader)


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
    
    # Model
    model = DiffusionModel(
        image_size=config['data']['image_size'],
        timesteps=config['model']['diffusion']['timesteps'],
        beta_start=config['model']['diffusion']['beta_start'],
        beta_end=config['model']['diffusion']['beta_end'],
        unet_channels=config['model']['diffusion']['unet_channels']
    ).to(device)
    
    model.register_schedule(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate_g'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
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
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
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
        loss = train_epoch(model, dataloader, optimizer, device, epoch, config, writer)
        
        print(f"Epoch {epoch}: Loss={loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, 
                      os.path.join(config['output']['checkpoint_dir'], f'diffusion_checkpoint_epoch_{epoch+1}.pth'))
            
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = model.p_sample_loop(
                    (4, 3, current_size, current_size),
                    device
                )
                samples = (samples + 1) / 2  # Denormalize
                
                from torchvision.utils import save_image
                save_image(samples, 
                          os.path.join(config['output']['sample_dir'], f'diffusion_samples_epoch_{epoch+1}.png'),
                          nrow=2)
    
    writer.close()


if __name__ == '__main__':
    main()
