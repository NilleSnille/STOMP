from typing import Any
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass
class MetaConfig:
    seed: int
    ckpt_freq: int = 0 # 0 means disabled

def load_meta_config(node_or_path: Any) -> DictConfig:
    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    schema = OmegaConf.structured(MetaConfig)
    merged_config = OmegaConf.merge(schema, node)
    return merged_config