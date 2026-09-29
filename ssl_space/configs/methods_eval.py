from typing import Any, Dict, Type
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass 
class BaseMethodEvalConfig:
    name: str
    finetune: bool
    normalize: bool
    validation_freq: int
    pretrain_dir: str

def load_method_eval_config(node_or_path: Any) -> DictConfig:

    _METHOD_EVAL_SCHEMAS: Dict[str, Type[BaseMethodEvalConfig]] = {
        "base_linear": BaseMethodEvalConfig,
    }

    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})

    method_name = str(node.get("name", "")).lower()
    if method_name not in _METHOD_EVAL_SCHEMAS:
        raise ValueError(
            f"Unknown or missing 'name': {method_name}. "
            f"Expected one of {list(_METHOD_EVAL_SCHEMAS)}"
        )
    
    schema = OmegaConf.structured(_METHOD_EVAL_SCHEMAS[method_name])
    merged_config = OmegaConf.merge(schema, node)
    return merged_config