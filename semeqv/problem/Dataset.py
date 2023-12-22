from abc import abstractmethod
import torch

class Dataset(torch.utils.data.Dataset):
    
    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def __len__(self) -> int: pass

    @abstractmethod
    def __getitem__(self, idx) -> dict: pass

    @abstractmethod
    def todevice(self, *args) -> dict: pass

    @abstractmethod
    def collate_fn(self, data) -> dict: pass

    def save(self): pass
    def restore(self, state): pass
