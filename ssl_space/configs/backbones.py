from typing import Tuple, Dict, Any, Type, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


@dataclass
class Dropped_DST_ViTArchKwargs:
    img_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int
    num_cls_tokens: int
    mlp_ratio: float
    qkv_bias: bool
    qk_scale: float|None
    feat_drop_p: float
    attn_drop_p: float
    path_drop_p: float
    dropout_p: float
    act_layer: str
    norm_layer: str
    num_frames: int
    interpolation_mode_space: str
    interpolation_mode_time: str


@dataclass
class BaseBackboneConfig:
    name: str
    arch_kwargs: Any
    pretrained_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Dropped_DST_ViTConfig(BaseBackboneConfig):
    arch_kwargs: Dropped_DST_ViTArchKwargs


def load_backbone_config(node_or_path: Any) -> DictConfig:
    
    _BACKBONE_SCHEMAS: Dict[str, Type[BaseBackboneConfig]] = {
        'dropped_dst_vit': Dropped_DST_ViTConfig,
    }

    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})

    backbone_name = str(node.get("name", "")).lower()
    if backbone_name not in _BACKBONE_SCHEMAS:
        raise ValueError(
            f"Unknown or missing backbone 'name': {backbone_name}. "
            f"Expected one of {list(_BACKBONE_SCHEMAS)}"
        )
    schema = OmegaConf.structured(_BACKBONE_SCHEMAS[backbone_name])
    merged_config = OmegaConf.merge(schema, node)
    return merged_config
