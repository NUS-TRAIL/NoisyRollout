import math
import torch
import numpy as np
import random
from PIL import Image
import torchvision.transforms as T
from ..workers.actor.config import ActorConfig

class ImageAugmenter:
    def __init__(self, config: ActorConfig=None):
        self.config = config or {}
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        
        # Set default configurations
        self.aug_type = self.config.aug_type
        self.noise_step = self.config.gaussian_noise_step  # Gaussian noise steps
        self.crop_size = self.config.crop_size  # Random occlusion region ratio
        self.rotate_angle = self.config.rotate_angle  # Maximum rotation angle
        self.decay_sig_mid_step = self.config.decay_sig_mid_step
        
        # Augmentation method mapping
        self.aug_methods = {
            'gaussian': self.apply_gaussian_noise,
            'crop_fill': self.apply_crop_fill,
            'rotate': self.apply_rotation,
        }

    def augment(self, image, step=0, total_steps=1):            
        # Handle decay
        decay = 1.0
        if hasattr(self.config, 'decay_mode') and hasattr(self.config, 'decay_coef'):
            decay_mode = self.config.decay_mode
            decay_coef = self.config.decay_coef
            norm_step = step / total_steps
            
            if decay_mode == 'exp':
                decay = 1.0 - decay_coef ** (total_steps - step)
            elif decay_mode == 'pow':
                decay = 1.0 - norm_step ** decay_coef
            elif decay_mode == 'linear':
                decay = 1.0 - norm_step
            elif decay_mode == 'sigmoid':
                x = decay_coef * (norm_step - self.decay_sig_mid_step / total_steps)
                decay = 1.0 - (1 / (1 + math.exp(-x)))
                
        aug_method = self.aug_methods.get(self.aug_type)
        if aug_method is None:
            return image
            
        return aug_method(image, decay)
        
    def apply_gaussian_noise(self, image, decay=1.0):
        """Apply gaussian noise to image"""
        image_tensor = self.to_tensor(image)
        noise_step = int(self.noise_step * decay)
        noisy_tensor = self._add_gaussian_noise(image_tensor, noise_step)
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        return self.to_pil(noisy_tensor)
    
    def _add_gaussian_noise(self, image_tensor, noise_step):
        """Implementation of gaussian noise"""
        num_steps = 1000
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        
        def q_x(x_0, t):
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t * x_0 + alphas_1_m_t * noise)
        
        return q_x(image_tensor, noise_step)
    
    def apply_rotation(self, image, decay=1.0):
        """Apply rotation to image"""
        max_angle = self.rotate_angle * decay
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    def apply_crop_fill(self, image, decay=1.0):
        """Apply random region filling (occlusion)"""
        width, height = image.size
        crop_size_scaled = self.crop_size * decay
        crop_w = int(width * crop_size_scaled)
        crop_h = int(height * crop_size_scaled)
        
        if width > crop_w and height > crop_h:
            x = random.randint(0, width - crop_w)
            y = random.randint(0, height - crop_h)
        else:
            x, y = 0, 0
            crop_w = min(crop_w, width)
            crop_h = min(crop_h, height)
            
        img_np = np.array(image)
        img_np[y:y+crop_h, x:x+crop_w] = 0
        
        return Image.fromarray(img_np)


def augment_images(images, config, step=0, total_steps=1):
    augmenter = ImageAugmenter(config)
    return [augmenter.augment(img, step, total_steps) for img in images]


def augment_batch(batch, config, step=0, total_steps=1):
    if "multi_modal_data" not in batch.non_tensor_batch:
        return batch
        
    from copy import deepcopy
    new_batch = deepcopy(batch)
    
    for i, item in enumerate(new_batch.non_tensor_batch["multi_modal_data"]):
        if "image" in item:
            image_list = item["image"]
            augmented_images = augment_images(
                image_list, 
                config, 
                step, 
                total_steps
            )
            new_batch.non_tensor_batch["multi_modal_data"][i]["image"] = augmented_images
            
    return new_batch