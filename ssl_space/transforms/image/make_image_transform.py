from typing import Tuple
from omegaconf import DictConfig
import random
from torchvision import transforms
from PIL import ImageOps, ImageFilter

class Solarization:
    def __call__(self, img):
        return ImageOps.solarize(img)
    
class GaussianBlur:
    def __init__(self, sigma = None):
        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class Equalization:
    def __call__(self, img):
        return ImageOps.equalize(img)

def make_base_transforms(dset_config: DictConfig) -> Tuple[transforms.ToTensor,transforms.Normalize]:    
    return (
        transforms.ToTensor(),
        transforms.Normalize(mean=dset_config.normalization.mean, std=dset_config.normalization.std),
    )

def make_composed_transform(dset_cfg: DictConfig, view_cfg: DictConfig):
    ops = []

    if view_cfg.resize is not None:
        ops.append(
            transforms.Resize(
                size=(view_cfg.resize.size, view_cfg.resize.size),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        )
    
    if view_cfg.center_crop is not None:
        ops.append(
            transforms.CenterCrop(
                size=view_cfg.center_crop.size,
            )
        )

    if view_cfg.rrc is not None:
        ops.append(
            transforms.RandomResizedCrop(
                size=view_cfg.rrc.size,
                scale=(view_cfg.rrc.min_scale, view_cfg.rrc.max_scale), 
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        )
        
    if view_cfg.color_jitter is not None:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=view_cfg.color_jitter.brightness,
                        contrast=view_cfg.color_jitter.contrast,
                        saturation=view_cfg.color_jitter.saturation,
                        hue=view_cfg.color_jitter.hue,
                    )
                ],
                p=view_cfg.color_jitter.prob
            )
        )

    if view_cfg.grayscale is not None:
        ops.append(transforms.RandomGrayscale(p=view_cfg.grayscale.prob))
    
    if view_cfg.gaussian_blur is not None:
        ops.append(transforms.RandomApply([GaussianBlur()], p=view_cfg.gaussian_blur.prob))
    
    if view_cfg.solarization is not None:
        ops.append(transforms.RandomApply([Solarization()], p=view_cfg.solarization.prob))
    
    if view_cfg.equalization is not None:
        ops.append(transforms.RandomApply([Equalization()], p=view_cfg.equalization.prob))
    
    if view_cfg.horizontal_flip is not None:
        ops.append(transforms.RandomHorizontalFlip(p=view_cfg.horizontal_flip.prob))
    
    ops.extend(make_base_transforms(dset_config=dset_cfg))
    ops = transforms.Compose(ops)
    return ops