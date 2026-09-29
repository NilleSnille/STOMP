from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

@dataclass
class TemporalAugOffsetConfig:
    start_offset: int
    end_offset: int

@dataclass
class TemporalAugRandomConfig:
    num_windows: int

@dataclass
class TemporalAugConfig:
    T: int
    offset: Optional[TemporalAugOffsetConfig] = None
    random: Optional[TemporalAugRandomConfig] = None

@dataclass
class Resize:
    size: int

@dataclass
class CenterCrop:
    size: int

@dataclass
class RandomCrop:
    size: int

@dataclass
class RandomResizedCroppingConfig:
    size: int
    min_scale: float
    max_scale: float

@dataclass
class HorizontalFlipConfig:
    prob: float

@dataclass
class ColorJitteringConfig:
    prob: float
    brightness: float
    contrast: float
    saturation: float
    hue: float|None = None

@dataclass
class GrayScaleConfig:
    prob: float

@dataclass
class SolarizationConfig:
    prob: float

@dataclass
class GaussianBlurConfig:
    prob: float

@dataclass
class EqualizationConfig:
    prob: float

@dataclass
class ColorNormalizeConfig:
    do: bool

@dataclass
class RandomShortSideScaleJitter:
    min_size: int
    max_size: int
    inverse_uniform_sampling: bool

@dataclass 
class ViewConfig:
    
    is_small: bool
    
    num_crops: int

    temporal: Optional[TemporalAugConfig] = None

    resize: Optional[Resize] = None

    center_crop: Optional[CenterCrop] = None
    
    rrc: Optional[RandomResizedCroppingConfig] = None

    random_short_side_scale_jitter: Optional[RandomShortSideScaleJitter] = None

    random_crop: Optional[RandomCrop] = None
    
    horizontal_flip: Optional[HorizontalFlipConfig] = None
    
    color_jitter: Optional[ColorJitteringConfig] = None
    
    grayscale: Optional[GrayScaleConfig] = None

    gaussian_blur: Optional[GaussianBlurConfig] = None

    solarization: Optional[SolarizationConfig] = None

    equalization: Optional[EqualizationConfig] = None

    color_normalize: Optional[ColorNormalizeConfig] = None

@dataclass
class AugmentationConfig:
    stage_type: str
    modality: str
    views: Dict[int, ViewConfig] = field(default_factory=dict)

    @property
    def num_crops(self) -> int:
        return sum(v.num_crops for v in self.views.values())

    @property
    def num_large_crops(self) -> int:
        return sum(v.num_crops for v in self.views.values() if not v.is_small)

    @property
    def num_small_crops(self) -> int:
        return sum(v.num_crops for v in self.views.values() if v.is_small)

def load_aug_config(node_or_path: Any) -> DictConfig:
    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})

    schema = OmegaConf.structured(AugmentationConfig)
    merged_config = OmegaConf.merge(schema, node)
    return merged_config