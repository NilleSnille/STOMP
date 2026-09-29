from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue

from .augmentations         import AugmentationConfig,      load_aug_config
from .backbones             import BaseBackboneConfig,      load_backbone_config
from .datasets              import BaseDatasetConfig,       load_dataset_config
from .meta                  import MetaConfig,              load_meta_config
from .methods               import BaseMethodConfig,        load_method_config
from .momentum              import MomentumConfig,          load_momentum_config
from .online_evaluations    import OnlineEvaluationConfig,  load_onlineeval_config
from .optimizers            import BaseOptimizerConfig,     load_optimizer_config
from .performance           import PerformanceConfig,       load_performance_config
from .lr_schedulers         import LRSchedulerBaseConfig,   load_lrscheduler_config
from .wd_schedulers         import WDSchedulerBaseConfig,   load_wdscheduler_config

from .methods_eval          import BaseMethodEvalConfig,    load_method_eval_config

@dataclass
class FullConfig:

    meta: MetaConfig

    performance: PerformanceConfig

    method: BaseMethodConfig
    
    backbone: BaseBackboneConfig

    optimizer: BaseOptimizerConfig
    
    lr_scheduler: LRSchedulerBaseConfig

    wd_scheduler: WDSchedulerBaseConfig
        
    dataset_train: BaseDatasetConfig
    
    augmentation_train: AugmentationConfig

    online_eval: OnlineEvaluationConfig

    dataset_eval: Optional[BaseDatasetConfig] = None

    augmentation_eval: Optional[AugmentationConfig] = None

    momentum: Optional[MomentumConfig] = None

    momentum_backbone: Optional[BaseBackboneConfig] = None

_SECTION_LOADERS = {

    "meta":                 load_meta_config,

    "performance":          load_performance_config,

    "method":               load_method_config,

    "backbone":             load_backbone_config,

    "optimizer":            load_optimizer_config,

    "lr_scheduler":         load_lrscheduler_config,

    "wd_scheduler":         load_wdscheduler_config,

    "dataset_train":        load_dataset_config,

    "augmentation_train":   load_aug_config,

    "online_eval":          load_onlineeval_config,
    
    "dataset_eval":         load_dataset_config,

    "augmentation_eval":    load_aug_config,
    
    "method_eval":          load_method_eval_config,

    "momentum":             load_momentum_config,

    "momentum_backbone":    load_backbone_config,
}

def _validate_cross_section(cfg: DictConfig) -> None:
    """
    Raise if the config has an illegal combination of high-level combinations.
    Should be extended with warning on lower level combinations. 
    """
    do_online_eval = cfg.online_eval.linear_eval or cfg.online_eval.knn_eval
    if do_online_eval:
        # eval dataset and eval augmentaion must exist
        if cfg.dataset_eval is None or cfg.augmentation_eval is None:
            raise ValueError(
                "Online evaluation enabled, so both dataset_eval and augmentation_eval are required."
            )
    else:
        # neither dataset_eval or augmentation_eval should be supplied
        if cfg.dataset_eval is not None or cfg.augmentation_eval is not None:
            raise ValueError(
                "No online evaluation to be performed. Omitt dataset_eval and augmentation_eval"
            )
        
    if cfg.momentum is not None: 
        assert cfg.momentum_backbone is not None
    else:
        assert cfg.momentum_backbone is None

def load_full_config(path: str) -> DictConfig:
    raw = OmegaConf.load(path)
    for key, loader in _SECTION_LOADERS.items():
        if key in raw:
            raw[key] = loader(raw[key])
    
    schema = OmegaConf.structured(FullConfig)
    merged_config = OmegaConf.merge(schema, raw)
    missing = OmegaConf.missing_keys(merged_config)
    if missing:
        raise MissingMandatoryValue(
            f"Configuration is missing mandatory values: {sorted(missing)}"
        )
    OmegaConf.resolve(merged_config)
    OmegaConf.set_readonly(merged_config, True)
    _validate_cross_section(merged_config)
    return merged_config


@dataclass
class FullConfigEval:

    meta: MetaConfig

    method_eval: BaseMethodEvalConfig
    
    optimizer: BaseOptimizerConfig
    
    lr_scheduler: LRSchedulerBaseConfig

    wd_scheduler: WDSchedulerBaseConfig
    
    performance: PerformanceConfig
    
    dataset_train: BaseDatasetConfig
    
    augmentation_train: AugmentationConfig
    
    backbone: BaseBackboneConfig

    dataset_eval: Optional[BaseDatasetConfig] = None

    augmentation_eval: Optional[AugmentationConfig] = None  

def load_full_config_eval(path: str) -> DictConfig:
    raw = OmegaConf.load(path)
    for key, loader in _SECTION_LOADERS.items():
        if key in raw:
            raw[key] = loader(raw[key])
    
    schema = OmegaConf.structured(FullConfigEval)
    merged_config = OmegaConf.merge(schema, raw)
    missing = OmegaConf.missing_keys(merged_config)
    if missing:
        raise MissingMandatoryValue(
            f"Configuration is missing mandatory values: {sorted(missing)}"
        )
    OmegaConf.resolve(merged_config)
    OmegaConf.set_readonly(merged_config, True)
    return merged_config
