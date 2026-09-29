from typing import Any
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass
class OnlineEvaluationConfig:
    evaluation_freq: int
    
    knn_eval: bool
    knn_k: int
    knn_temperature: float
    knn_distance_func: str

    linear_eval_lr: float    
    linear_eval: bool
    linear_eval_momentum: bool = False

def load_onlineeval_config(node_or_path: Any) -> DictConfig:
    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    schema = OmegaConf.structured(OnlineEvaluationConfig)
    merged_config = OmegaConf.merge(schema, node)
    return merged_config