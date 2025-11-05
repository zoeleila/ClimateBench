
import glob
import xarray as xr
import pandas as pd
import torch
import argparse
import numpy as np
from torchvision.transforms import v2
from pathlib import Path
import os

from ClimateBench.funcs.lightning_module import ClimateBenchLightningModule
from ClimateBench.funcs.transforms import ToTensor, Normalize
from ClimateBench.bin.preprocessing.build_dataset import BuilderDataset
from ClimateBench.funcs.settings import PREDICTIONS_DIR



def get_target_format(simulation_name: str, config: dict):
    var_to_predict = config['model']['var_to_predict']
    builder = BuilderDataset(raw_dir=Path(config['data']['raw_dir']),
                             slider=config['data']['slider'],
                             len_historical=config['data']['len_historical'])
    output_xr = builder.load_output(simulation_name)
    shape = tuple(output_xr.dims[d] for d in output_xr.coords)
    output_xr = output_xr.drop_vars(output_xr.keys())
    new_xr = output_xr.assign({var_to_predict : (tuple(output_xr.coords), np.empty(shape))})
    return new_xr

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict and plot results for full period")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_simu' , type=str, default=None, help='Simulation to test on')
    args = parser.parse_args()

    hparams_file = Path(args.checkpoint).parent.parent / 'hparams.yaml'
    device = 'cpu'
    model = ClimateBenchLightningModule.load_from_checkpoint(args.checkpoint, map_location=device, hparams_file = hparams_file)
    model.eval()
    config = model.hparams['config']
    sample_dir = Path(config['data']['dataset_dir'])
    var_to_predict = config['model']['var_to_predict']
    exp = config['data']['exp']
    test_name = config['model']['test_name']
    print(test_name)

    if args.test_simu is None:
        simu = config['model']['simus_test'][0]
    else:
        simu = args.test_simu

    transforms = v2.Compose([
                ToTensor(),
                Normalize(sample_dir)
                ])
     
    list_samples = np.sort(glob.glob(str(sample_dir/f'sample_{simu}_*.npz')))
    start_date = list_samples[0].split('_')[-1].split('.npz')[0]
    end_date = list_samples[-1].split('_')[-1].split('.npz')[0]
    ds = get_target_format(simu, config)
    ds = ds.sel(time=slice(start_date, end_date))

    for i, sample in enumerate(list_samples):
        print(sample)
        data = dict(np.load(sample), allow_pickle=True)

        x = data['x']
        y = np.random.randn(1, 1, x.shape[2], x.shape[3]) 
        x, y = transforms((x, y))
        x = torch.unsqueeze(x, dim=0).float()
        y_hat = model(x.to(device)).to(device)
        y_hat = y_hat.detach().cpu()
        y_hat = torch.squeeze(y_hat)

        if ((var_to_predict == "pr90") | (var_to_predict == "pr")):
            y_hat / 86400.
        ds[var_to_predict][i] = y_hat.numpy()

    ds.to_netcdf(PREDICTIONS_DIR/ exp / f'{simu}_{start_date}_{end_date}_{test_name}.nc')
    