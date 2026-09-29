from typing import Any
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass
class MomentumConfig:
    base_tau: float
    final_tau: float
    use_eman: bool = False

def load_momentum_config(node_or_path: Any) -> DictConfig:
    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    schema = OmegaConf.structured(MomentumConfig)
    merged_config = OmegaConf.merge(schema, node)
    return merged_config