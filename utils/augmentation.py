import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2


def elastic_transform(image, alpha=100, sigma=10):
    """Elastic deformation of images."""
    image_np = np.array(image)
    shape = image_np.shape
    dx = np.random.randn(*shape[:2]) * alpha
    dy = np.random.randn(*shape[:2]) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted = cv2.remap(
        image_np,
        np.float32(np.reshape(x + dx, shape[:2])),
        np.float32(np.reshape(y + dy, shape[:2])),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return Image.fromarray(distorted)


class ElasticTransform:
    """Wrapper for elastic transform."""
    def __init__(self, alpha=100, sigma=10, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img):
        if np.random.rand() < self.p:
            return elastic_transform(img, self.alpha, self.sigma)
        return img


def get_augmentation_transform(jitter_brightness=0.2, jitter_contrast=0.2,
                               jitter_saturation=0.2, max_rotation=15,
                               use_elastic=False):
    """Create augmentation transform pipeline."""
    transform_list = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=0.1
        ),
        transforms.RandomRotation(degrees=max_rotation),
    ]
    
    if use_elastic:
        transform_list.append(ElasticTransform(p=0.3))
    
    return transforms.Compose(transform_list)
