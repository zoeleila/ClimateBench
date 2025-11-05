import numpy as np
import torch
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path
import re
import json
import glob

from ClimateBench.funcs.plotutils import plot_histogram

def compute_stats(list_samples, config:dict)->dict:
    channels = config['data']['vars']
    ch = len(channels)
    sum = torch.zeros(ch)
    square_sum = torch.zeros(ch)
    n_total = torch.zeros(ch)

    init_sample = dict(np.load(list_samples[0], allow_pickle=True))
    x_init = init_sample['x']
    x_init = torch.tensor(x_init[-1:,:,:,:])
    x_init = x_init.flatten(start_dim=0, end_dim=3) 
    min = torch.min(x_init, dim=0).values
    max = torch.max(x_init, dim=0).values

    for sample in tqdm(list_samples, desc="Computing stats from dataset"):
        x = dict(np.load(sample, allow_pickle=True))['x']
        x = torch.tensor(x[-1:,:,:,:])  # Take only the last time step because we don't want redondancy
        n = x.shape[0] * x.shape[1] * x.shape[2]  # time * height * width
        x = x.flatten(start_dim=0, end_dim=2)  # shape = (last time step * height * width, channels)
        sum += torch.sum(x, dim=(0))
        square_sum += torch.sum(x**2, dim=(0))
        n_total += n
        min = torch.min(min, torch.min(x, dim=0).values)
        max = torch.max(max, torch.max(x, dim=0).values)
    
    list_samples_hist_start = [f for f in list_samples if '_1859' in f]
    for sample in tqdm(list_samples_hist_start, desc="Computing stats from historical period"):
        x = dict(np.load(sample, allow_pickle=True))['x']
        x = torch.tensor(x[:-1,:,:,:])  # Take only the 9 first time steps
        n = x.shape[0] * x.shape[1] * x.shape[2]  # time * height * width
        x = x.flatten(start_dim=0, end_dim=2)  # shape = (last time step * height * width, channels)
        sum += torch.sum(x, dim=(0))
        square_sum += torch.sum(x**2, dim=(0))
        n_total += n
        min = torch.min(min, torch.min(x, dim=0).values)
        max = torch.max(max, torch.max(x, dim=0).values)
    print(n_total)
    mean = sum / n_total
    std = torch.sqrt((square_sum / n_total) - (mean ** 2))
    stats = {}
    for i, chanel in enumerate(channels):
        stats[chanel] = {'mean': mean[i].item(),
                         'std': std[i].item(),
                         'min': min[i].item(),
                         'max': max[i].item()}
    print(stats)
    return stats

def compute_stats2(list_samples, config:dict, method='3')->dict:
    channels = config['data']['vars']
    x_full = []
    for samples in tqdm(list_samples, desc="Loading all data into memory"):
        x = dict(np.load(samples, allow_pickle=True))['x']
        if '_1859' in samples:
            x_full.append(x[:,:,:,:])  # Take only the 9 first time steps
        else:
            x_full.append(x[-1:,:,:,:])  # Take only the last time step because we don't want redondancy
    x_full = np.concatenate(x_full, axis=0)
    stats = {}
    for i, channel in enumerate(channels):
        array = x_full[:,:,:,i]
        stats[channel] = {'mean': array.mean(),
                        'std': array.std(),
                        'min': array.min(),
                        'max': array.max()}
    print(stats)
    config_plot_path = Path(config_path).parent / 'config_plots.yaml'
    with open(config_plot_path) as file:
        config_plot = yaml.safe_load(file)    
    plot_histogram(x, 
                   stats, 
                   title='Data histograms used to compute statistics', 
                   save_dir=Path(config['data']['dataset_dir']) / f'histogram{method}.png', 
                   xlim_dict=None)

    return stats


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--histograms', type=bool, default=False, help='Whether to plot histograms or not')
    parser.add_argument('--method', type=str, default='3', help='Method to calculate statistics (1=paper, 2=val+train, 3=train)')
    args = parser.parse_args()
    config_path = args.config
    method = args.method
    histograms = args.histograms

    with open(config_path) as file:
        config = yaml.safe_load(file)

    sample_dir = Path(config['data']['dataset_dir'])
    list_samples = list(glob.glob(str(sample_dir / f'sample*.npz'))) # train and val as in ClimateBench
    list_simu = [f'_{simu}_' for simu in config['model']['simus_train']]
    regex = re.compile('|'.join(re.escape(p) for p in list_simu))
    list_samples = [f for f in list_samples if regex.search(f)]

    if method == '1': 
        # Like in paper computing 3 times historical statistics
        list_samples_historical = [f for f in list_samples if '_historical_' in f]
        list_samples.extend(list_samples_historical)
        list_samples.extend(list_samples_historical) # two times for ssp126 and ssp370
    if method == '3':
        # Correct way
        stop = 0.2
        np.random.seed(6)
        np.random.shuffle(list_samples)
        split_index = int(len(list_samples) * (1 - stop))
        list_samples = list_samples[:split_index]
        print(list_samples)

    if histograms:
        stats = compute_stats2(list_samples, config, method)
    else:
        stats = compute_stats(list_samples, config)
    with open(sample_dir/f'statistics{method}.json', "w") as f: 
        json.dump(stats, f)