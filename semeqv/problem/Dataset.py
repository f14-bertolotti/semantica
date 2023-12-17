from abc import abstractmethod
import torch

class Dataset(torch.utils.data.Dataset):
    
    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __getitem__(self, idx): pass

    @abstractmethod
    def todevice(self): pass

    @abstractmethod
    def collate_fn(self, data): pass

