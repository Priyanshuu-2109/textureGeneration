"""
Lightweight transforms using PIL + torch only. Use when torchvision fails
(e.g. missing _lzma in pyenv Python on macOS).
"""
import random
import torch
import numpy as np
from PIL import Image, ImageEnhance


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR)


class ToTensor:
    def __call__(self, img):
        arr = np.array(img)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).float() / 255.0


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        if self.hue > 0:
            # PIL doesn't have hue - skip or approximate with Color
            pass
        return img


class RandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
