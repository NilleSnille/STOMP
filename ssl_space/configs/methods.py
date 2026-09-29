from typing import Any, Dict, Type
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

@dataclass 
class BaseMethodConfig:
    name: str

@dataclass
class STOMP(BaseMethodConfig):
    proj_num_layers: int
    proj_use_bn: bool
    proj_norm_last_layer: bool
    proj_hidden_dim: int
    proj_bottleneck_dim: int
    proj_output_dim: int

    # dino loss
    inv_warmup_teacher_temp: float
    inv_teacher_temp: float
    inv_warmup_teacher_temp_epochs: int
    inv_student_temp: float
    inv_center_momentum: float

    # Dynamics
    freeze_last_layer_epochs: int

    # Masking
    global_mask_ratio: float
    global_mask_candidate_ratio: float

    # sinkhorn
    num_prototypes: int
    sinkhorn_eps: float
    sinkhorn_iters: int
    sinkhorn_logits_temp: float
    sinkhorn_normalise_target: bool
    sinkhorn_soft_targets: bool

    # decoder
    use_temporal_decoder: bool

    # component strength
    recon_strength: float
    invariance_strength: float

def load_method_config(node_or_path: Any) -> DictConfig:

    _METHOD_SCHEMAS: Dict[str, Type[BaseMethodConfig]] = {
        "stomp": STOMP,
    }

    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})

    method_name = str(node.get("name", "")).lower()
    if method_name not in _METHOD_SCHEMAS:
        raise ValueError(
            f"Unknown or missing 'name': {method_name}. "
            f"Expected one of {list(_METHOD_SCHEMAS)}"
        )
    
    schema = OmegaConf.structured(_METHOD_SCHEMAS[method_name])
    merged_config = OmegaConf.merge(schema, node)
    return merged_config