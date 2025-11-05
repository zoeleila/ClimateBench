from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import glob
from typing import Optional
import torch
from torch import Tensor
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
import re

from ClimateBench.funcs.transforms import ToTensor, Normalize


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
        self.vars_to_predict = config['data']['vars_to_predict']
        self.var_to_predict = config['model']['var_to_predict']

        stop = 0.2
        list_samples = np.sort(glob.glob(str(self.sample_dir / f'sample*.npz')))
        
        if self.data_type == 'test':
            list_simu = [f'_{simu}_' for simu in self.simus_test]
            regex = re.compile('|'.join(re.escape(p) for p in list_simu))
            list_samples = [f for f in list_samples if regex.search(f)]
            ordre_exp = {exp: i for i, exp in enumerate(self.simus_test)}
            list_samples = sorted(
                list_samples,
                key=lambda f: (
                    ordre_exp.get(Path(f).stem.split('_')[1], float('inf'))
                )
            )
            self.samples = list_samples
        else:
            list_simu = [f'_{simu}_' for simu in self.simus_train]
            regex = re.compile('|'.join(re.escape(p) for p in list_simu))
            list_samples = [f for f in list_samples if regex.search(f)]
            ordre_exp = {exp: i for i, exp in enumerate(self.simus_train)}
            list_samples = sorted(
                list_samples,
                key=lambda f: (
                    ordre_exp.get(Path(f).stem.split('_')[1], float('inf'))
                )
            )
            
            np.random.seed(6)
            #np.random.shuffle(list_samples) #A ajouter pour la manière correct de mélanger les données
            split_index = int(len(list_samples) * (1 - stop))
            if self.data_type == 'train':
                self.samples = list_samples[:split_index]
            else: 
                self.samples = list_samples[split_index:]

    def get_ivar_to_predict(self) -> int:
        """
        Returns the index of the variable to predict based on its name.
        """
        return self.vars_to_predict.index(self.var_to_predict)


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
        x, y = data['x'], data['y'][:,:,:,self.get_ivar_to_predict()]
        if self.transform:
            x, y = self.transform((x, y))
            x.float(), y.float()
        return x, y 


def get_dataloaders(data_type: str, config:dict, transforms:bool=True) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the specified data type.
    Args:
        data_type (str): The type of data to load. Expected values are 'train' or other types
                            (e.g., 'validation', 'test'). Determines the shuffle behavior and batch size.
    Returns:
        DataLoader: A PyTorch DataLoader object configured with the appropriate dataset,
                    transformations, batch size, and shuffle settings.
    """
    if transforms:
        transforms = v2.Compose([ToTensor(),
                                 Normalize(dataset_dir=Path(config['data']['dataset_dir']))])
    else:
        transforms = v2.Compose([ToTensor()])

    if data_type == 'train':
        shuffle = True
    else:
        shuffle = False
    
    dataset = ClimateBench(transform=transforms,
                            config=config,
                            data_type=data_type)
    
    if data_type == 'train':
        batch_size = config['model']['batch_size']
    else : 
        batch_size = 1

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=1)
    return dataloader

def get_stats_from_dataloader(train_dataloader: DataLoader,
                              val_dataloader: DataLoader = None, 
                              config:dict=None) -> None:
    """
    Computes the mean and standard deviation of the input variable across the entire dataset (scalars !).
    x.shape = (batch_size, slider, height, width, channels) 
    Carefull, doesn't take the first 9 years from 1850 to 1859 into account, use compute_statistics.py script for that.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader object containing the dataset. 
        channels (list): List of channel names corresponding to the input variable.

    """
    channels = config['data']['vars']
    ch = len(channels)
    sum = torch.zeros(ch)
    square_sum = torch.zeros(ch)
    n_total = torch.zeros(ch)

    init_batch = next(iter(train_dataloader))
    x_init, _ = init_batch
    x_init = x_init[:,-1:,:,:,:] 
    x_init = x_init.flatten(start_dim=0, end_dim=3) 
    min = torch.min(x_init, dim=0).values
    max = torch.max(x_init, dim=0).values

    dataloaders = [train_dataloader, val_dataloader] if val_dataloader is not None else [train_dataloader]
    for dataloader in dataloaders:
        for batch in tqdm(dataloader, desc="Computing stats from dataloader"):
            x, _ = batch
            x = x[:,-1:,:,:,:]  # Take only the last time step because we don't want redondancy
            n = x.shape[0] * x.shape[2] * x.shape[3]  # batch_size * height * width
            x = x.flatten(start_dim=0, end_dim=3)  # shape = (batch_size, last time step, height * width, channels)
            sum += torch.sum(x, dim=(0))
            square_sum += torch.sum(x**2, dim=(0))
            n_total += n
            min = torch.min(min, torch.min(x, dim=0).values)
            max = torch.max(max, torch.max(x, dim=0).values)
    mean = sum / n_total
    std = torch.sqrt((square_sum / n_total) - (mean ** 2))
    print(f"Train - Mean: {mean}, Std: {std}")
    stats = {}
    for i, chanel in enumerate(channels):
        stats[chanel] = {'mean': mean[i],
                         'std': std[i],
                         'min': min[i],
                         'max': max[i]}

    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.safe_load(file)

    train_dataloader = get_dataloaders(data_type='train', config=config)
    for i, batch in enumerate(train_dataloader):
        if i==0:
            x, y = batch
            print('x shape:', x.shape)
            print('y shape:', y.shape)
        else:
            break
    