from typing import Tuple, Dict, Type, Any, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

@dataclass
class NormalizationConfig:
    mean: Tuple[float, float, float]  # (R, G, B)
    std: Tuple[float, float, float]   # (R, G, B)

@dataclass
class BaseDatasetConfig:
    name: str
    root: str
    split: str
    num_classes: int
    normalization: NormalizationConfig
    with_index: bool
    inference_batch_size: Optional[int] = field(default=None, kw_only=True)

@dataclass
class EndoFMConfig(BaseDatasetConfig):
    num_clips_from_video: int
    num_frames_from_clip: int
    sampling_rate: int
    target_fps: float
    multithread_decode: bool
    decoding_backend: str
    mode: str
    num_retries: int

@dataclass
class PolypDiag(BaseDatasetConfig):
    mode: str
    num_clips_from_video: int
    num_crops_from_clip: int
    num_frames_from_crop: int
    sampling_rate: int
    target_fps: float
    multithread_decode: bool
    decoding_backend: str
    num_retries: int

def load_dataset_config(node_or_path: Any) -> DictConfig:

    _DATASET_SCHEMAS: Dict[str,Type[BaseDatasetConfig]] = {
        "endofm": EndoFMConfig,
        "polypdiag": PolypDiag,
    }

    if isinstance(node_or_path, str):
        node = OmegaConf.load(node_or_path)
    else:
        node = OmegaConf.create(node_or_path or {})
    
    dset_name = str(node.get("name", "")).lower()
    if dset_name not in _DATASET_SCHEMAS:
        raise ValueError(
            f"Unknown or missing dataset 'name': {dset_name}. "
            f"Expected one of {list(_DATASET_SCHEMAS)}"
        )        

    schema = OmegaConf.structured(_DATASET_SCHEMAS[dset_name])
    merged_config = OmegaConf.merge(schema, node)
    return merged_config