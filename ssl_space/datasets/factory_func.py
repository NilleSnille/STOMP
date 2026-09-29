import inspect
from typing                 import Dict, Any, Type
from omegaconf              import DictConfig

from .base_dataset              import BaseDataset, DualDataset
from .endofm                    import EndoFM
from .polypdiag                 import PolypDiag

def dataset_with_index(DatasetClass: Type[BaseDataset]) -> Type[BaseDataset]:
    """ Creates a wrapper aound a dataset, which also returns the data index in __getitem__.
        Args: DatsetClass (Type[BaseDataset]): Datset class to be wrapped.
        Returns: Type[BaseDataset]: dataset class with index
    """
    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)
        
    return DatasetWithIndex

def _kwargs_for_class_from_cfg(cls: Type[BaseDataset], cfg: DictConfig) -> Dict[str,Any]:
    '''
    Build kwargs for cls by intersecting cfg keys with the __init__ signature
    '''
    params = inspect.signature(cls.__init__).parameters
    allowed = set(params.keys()) - {"self", "transform", "target_transform"}
    return {k: v for k, v in cfg.items() if k in allowed}

def make_dataset(dset_cfg: DictConfig, transform=None, target_transform=None):
    _DATSETS_MAP: Dict[str,Type[BaseDataset]] = {
        'endofm':                   EndoFM,
        'polypdiag':                PolypDiag,

    }

    name = dset_cfg.name
    if name not in _DATSETS_MAP: 
        raise ValueError(f"Unknown dataset name: {name}")
    
    dset_cls = _DATSETS_MAP[name]
    if dset_cfg.with_index:
        dset_cls = dataset_with_index(dset_cls)

    dset_kwargs = _kwargs_for_class_from_cfg(dset_cls, dset_cfg)

    dataset = dset_cls(transform=transform, target_transform=target_transform, **dset_kwargs)

    return dataset

def make_dual_dataset(main_dataset: BaseDataset, side_dataset: BaseDataset) -> DualDataset:
    """Creates a wrapper around two dataset. This is to allow a custom sampler to easily sample
       from the datasets according to some strategy, and to allow it to be used in training loops in the same 
       way any other dataset is, without needing to keep track of two separate datasets.

        Args:   DatsetClassMain (Type[BaseDataset]): Datset class to be wrapped.
                DatsetClassSide (Type[BaseDataset]): Datset class to be wrapped.
        Returns: DualDataset: combined dataset class
    """
    return DualDataset(main_dataset, side_dataset)