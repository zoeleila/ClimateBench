import torch
from torch import Tensor
import numpy as np
import json

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample: tuple[np.ndarray, np.ndarray]) -> tuple[Tensor, Tensor]:
        x, y = sample
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class Normalize:
    """Normalize a tensor sample with mean and standard deviation."""
    def __init__(self, dataset_dir):
        stats_path = dataset_dir / 'statistics3.json'
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = torch.tensor([stats[c]['mean'] for c in stats])
        self.std = torch.tensor([stats[c]['std'] for c in stats])

    def __call__(self, sample: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, y = sample
        x = [(x[..., c] - self.mean[c]) / self.std[c] for c in range(x.shape[-1])]
        x = torch.stack(x, dim=-1)
        return x, y