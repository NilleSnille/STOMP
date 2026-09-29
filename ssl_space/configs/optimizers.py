from typing import Tuple, Dict, Type, Any
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

# Base class for all optimizers
@dataclass
class BaseOptimizerConfig:
    name: str
    max_epochs: int
    batch_size: int
    
    lr: float
    weight_decay: float
    exclude_bias_n_norm_wd: bool
    clip_grad: float = -1.   # a positive value will clip the gradients at that value. Negative values means no clipping.
    clip_grad_local: bool = True # only relevant with clip_grad > 0
    scale_lr_linear: bool = True # scale the learning rate linearly with the global batch size. Base is BatchSize 256.
    backbone_lr_scale: float = 1.0
    backbone_wd_scale: float = 1.0

# specific optimizer configs
@dataclass
class SGDConfig(BaseOptimizerConfig):
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False

@dataclass
class AdamConfig(BaseOptimizerConfig):
    betas: Tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

@dataclass
class AdamWConfig(BaseOptimizerConfig):
    betas: Tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

@dataclass
class LarsConfig(BaseOptimizerConfig):
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False
    eta: float = 1e-3 # trust coefficient for computing lr
    eps: float = 1e-8 # eps for division denominator
    clip_lr: bool = False # this scales the gradients by the norm - so no "regular" clipping is done.
    exclude_bias_n_norm: bool = False # exludes bias and norms from LARS scaling -> updates with SGD.

def load_optimizer_config(node_or_path: Any) -> DictConfig:

    _OPTIMIZER_SCHEMAS: Dict[str, Type[BaseOptimizerConfig]] = {
        "sgd": SGDConfig,
        "adam": AdamConfig,
        "adamw": AdamWConfig,
        "lars": LarsConfig,
    }

    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    opt_name = str(node.get("name", "")).lower()
    if opt_name not in _OPTIMIZER_SCHEMAS:
        raise ValueError(
            f"Unknown or missing optimiser 'name': {opt_name}. "
            f"Expected one of {list(_OPTIMIZER_SCHEMAS)}"
        )
    schema = OmegaConf.structured(_OPTIMIZER_SCHEMAS[opt_name])
    merged_config = OmegaConf.merge(schema, node)
    return merged_config