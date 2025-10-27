import torch
from torch import Tensor
import numpy as np

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample: tuple[np.ndarray, np.ndarray]) -> tuple[Tensor, Tensor]:
        x, y = sample
        return torch.tensor(x), torch.tensor(y)