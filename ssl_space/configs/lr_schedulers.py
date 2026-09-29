from typing import Any
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class LRSchedulerBaseConfig:
    name: str
    interval: str
    lr_decay_steps: int | None

@dataclass
class LRSchedulerCosineConfig(LRSchedulerBaseConfig):
    end_lr: float
    warmup_start_lr: float
    warmup_epochs: float

def load_lrscheduler_config(node_or_path: Any) -> LRSchedulerBaseConfig:
    _SCHEMAS = {
        "constant": LRSchedulerBaseConfig,
        "cosine": LRSchedulerCosineConfig,
    }
    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    name = str(node.get("name", "")).lower()
    if name not in _SCHEMAS:
        raise ValueError(
            f"Unknown or missing wd_schduler 'name': {name}. "
            f"Expected one of {list(_SCHEMAS)}"
        )
    
    schema = OmegaConf.structured(_SCHEMAS[name])
    merged_config = OmegaConf.merge(schema, node)
    return merged_config