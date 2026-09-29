from typing import Any
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass
class PerformanceConfig:
    disable_channel_last: bool
    num_workers: int
    sync_batchnorm: bool    
    precision: str  # 32 or mixed-fp16

def load_performance_config(node_or_path: Any) -> DictConfig:
    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    schema = OmegaConf.structured(PerformanceConfig)
    merged_config = OmegaConf.merge(schema, node)
    return merged_config