from typing import Tuple, List, Any
from torch.utils.data import Dataset

import torch

class BaseDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __get_item__")
    
    def do_transform(self, img, target) -> Tuple[List[torch.Tensor],Any]:
        if self.transform: img = self.transform(img)
        if self.target_transform: target = self.target_transform(target)
        return img, target

class DualDataset(BaseDataset):
    def __init__(self, main_dataset: BaseDataset, side_dataset: BaseDataset):
        self.main_dataset = main_dataset
        self.side_dataset = side_dataset
        self.main_len = len(main_dataset)
        self.side_len = len(side_dataset)
        self.total_len = self.main_len + self.side_len

    def __getitem__(self, index: int):
        if index < self.main_len:
            return self.main_dataset.__getitem__(index)
        else:
            return self.side_dataset.__getitem__(index - self.main_len)

    def __len__(self):
        return self.total_len