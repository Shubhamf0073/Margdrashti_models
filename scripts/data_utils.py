import numpy as np
from PIL import Image, ImageEnhance
import random
import torch

def _rand_bool(p: float) -> bool:
    return random.random() < p

def augment_pil(img: Image.Image) -> Image.Image:
    if _rand_bool(0.5):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if _rand_bool(0.8):
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    
    if _rand_bool(0.5):
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
    
    return img

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2,0,1))
    t = torch.from_numpy(arr)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t - mean) / std
