import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from .augmentation import get_augmentation_transform


class TextureDataset(Dataset):
    """
    Dataset for texture images with heavy augmentation and patch sampling.
    Supports progressive resizing and overlapping patch extraction.
    """
    def __init__(self, data_dir, image_size=256, patch_size=64, 
                 augment=True, mode='train', patches_per_image=100):
        self.data_dir = data_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.augment = augment and (mode == 'train')
        self.mode = mode
        self.patches_per_image = patches_per_image
        
        # Load image paths
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(
                [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.lower().endswith(ext.replace('*', ''))]
            )
        
        self.image_paths = sorted(self.image_paths)
        
        # Base transform
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Augmentation transform
        if self.augment:
            self.aug_transform = get_augmentation_transform()
        
        # Pre-generate patches for efficiency
        self.patches = []
        self._generate_patches()
    
    def _generate_patches(self):
        """Pre-generate overlapping patches from all images."""
        for img_path in self.image_paths:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            # Generate overlapping patches
            stride = self.patch_size // 2  # 50% overlap
            for i in range(0, h - self.patch_size + 1, stride):
                for j in range(0, w - self.patch_size + 1, stride):
                    patch = img_array[i:i+self.patch_size, j:j+self.patch_size]
                    self.patches.append((patch, img_path))
            
            # Also add random patches
            for _ in range(self.patches_per_image):
                i = np.random.randint(0, max(1, h - self.patch_size))
                j = np.random.randint(0, max(1, w - self.patch_size))
                patch = img_array[i:i+self.patch_size, j:j+self.patch_size]
                self.patches.append((patch, img_path))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_array, img_path = self.patches[idx]
        patch = Image.fromarray(patch_array)
        
        # Apply transforms
        if self.augment:
            patch = self.aug_transform(patch)
        
        patch = self.base_transform(patch)
        
        return patch, img_path


def get_dataloader(data_dir, image_size=256, patch_size=64, batch_size=8,
                   augment=True, mode='train', num_workers=4, patches_per_image=100):
    """Create a DataLoader for texture dataset."""
    dataset = TextureDataset(
        data_dir=data_dir,
        image_size=image_size,
        patch_size=patch_size,
        augment=augment,
        mode=mode,
        patches_per_image=patches_per_image
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
