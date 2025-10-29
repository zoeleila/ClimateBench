import glob
import xarray as xr
import pandas as pd
import torch
import argparse
import numpy as np
from torchvision.transforms import v2
import yaml
from pathlib import Path
from tqdm import tqdm

from ClimateBench.funcs.lightning_module import ClimateBenchLightningModule
from ClimateBench.funcs.transforms import ToTensor, Normalize
from ClimateBench.funcs.dataloader import get_dataloaders

'''

def get_target_format(exp:str, dates):
    get_data = Data(CONFIG[exp]['domain'])
    ds_target = get_data.get_target_dataset(target = CONFIG[exp]['target'], 
                                            var = CONFIG[exp]['target_vars'][0],
                                            date=pd.Timestamp('2014-12-31 00:00:00'))
    y = ds_target.tas.values
    
    if 'x' in ds_target.dims:
        ds = xr.Dataset(
            data_vars={'tas': (['time', 'y', 'x'], np.empty((len(dates), y.shape[0], y.shape[1])))},
            coords={"time" : dates,
                        "y" : ds_target.y.values,
                        "x" : ds_target.x.values})
        if exp == 'exp3':
            y = remove_countries(y)
    elif 'lon' in ds_target.dims:
        ds = xr.Dataset(
            data_vars={'tas': (['time', 'lat', 'lon'], np.empty((len(dates), y.shape[0], y.shape[1])))},
            coords={"time" : dates,
                        "lat" : ds_target.lat.values,
                        "lon" : ds_target.lon.values})
    return ds, y
'''

def get_target_format(config):
    simus_test = config['model']['simus_test']
    var_to_predict = config['model']['var_to_predict']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict and plot results for full period")
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint

    device = 'cpu'
    model = ClimateBenchLightningModule.load_from_checkpoint(checkpoint_dir, map_location=device)
    model.eval()
    config = model.hparams['config']
    test_dataloader = get_dataloaders('test', config)
    
    #ds, y = get_target_format(args.exp, dates=dates)
    #y = np.expand_dims(y, axis= 0)

    y_full = []
    y_hat_full = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        x, y_true = batch
        x = x.float().to(device)
        with torch.no_grad():
            y_hat = model(x).cpu().numpy()    
        y_full.append(y_true.numpy())
        y_hat_full.append(y_hat)
    y_full = np.concatenate(y_full, axis=0)
    y_hat_full = np.concatenate(y_hat_full, axis=0)
    
    
    ds.to_netcdf(PREDICTION_DIR/f'tas_day_{data_type}_{period}_r1i1p1f2_gr_{startdate}_{enddate}_{args.exp}_{test_name}.nc')
    