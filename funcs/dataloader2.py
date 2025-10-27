from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import glob
from typing import Optional
from torch import Tensor
import argparse
import yaml
from pathlib import Path

from ClimateBench.funcs.transforms import ToTensor

class ClimateBench(Dataset):
    def __init__(self,
                 transform: Optional[v2.Compose],
                 config,
                 data_type: str = 'train'):
        self.transform = transform
        self.data_type = data_type
        self.sample_dir = Path(config['data']['dataset_dir'])
        self.simus_train = config['model']['simus_train']
        self.simus_test = config['model']['simus_test']

        stop = 0.2
        list_samples = glob.glob(str(self.sample_dir / f'sample*.npz'))
        print(list_samples)
        if self.data_type == 'test':
            self.samples = [f for f in list_samples if any(simu in f for simu in self.simus_test)]
        else:
            list_samples = [f for f in list_samples if any(simu in f for simu in self.simus_train)]
            split_index = int(len(list_samples) * (1 - stop))
            if self.data_type == 'train':
                self.samples = list_samples[:split_index]
            else: 
                self.samples = list_samples[split_index:]
        print(self.samples)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: Transformed input (x) and target (y) tensors.
        """
        data = dict(np.load(self.samples[idx], allow_pickle=True))
        x, y = data['x'], data['y']
        if self.transform:
            x, y = self.transform((x, y))
        return x.float(), y.float()


def get_dataloaders(data_type: str, config:dict) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the specified data type.
    Args:
        data_type (str): The type of data to load. Expected values are 'train' or other types
                            (e.g., 'validation', 'test'). Determines the shuffle behavior and batch size.
    Returns:
        DataLoader: A PyTorch DataLoader object configured with the appropriate dataset,
                    transformations, batch size, and shuffle settings.
    """

    transforms = v2.Compose([ToTensor])
    training_data = ClimateBench(transform=transforms,
                            config=config,
                            data_type=data_type)
    
    
    if data_type == 'train':
        shuffle = True
    else:
        shuffle = False

    if data_type == 'train':
        batch_size = config['model']['batch_size']
    else : 
        batch_size = 1

    dataloader = DataLoader(training_data, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=1)
    return dataloader  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.safe_load(file)

    dataloader = get_dataloaders(data_type='train', config=config)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        break