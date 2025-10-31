
import glob
import xarray as xr
import pandas as pd
import torch
import argparse
import numpy as np
from torchvision.transforms import v2

from ClimateBench.funcs.lightning_module import ClimateBenchLightningModule
from ClimateBench.funcs.transforms import ToTensor, Normalize

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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict and plot results for full period")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    device = 'cpu'
    model = ClimateBenchLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    config = model.hparams['config']
    sample_dir = config['data']['dataset_dir']

    transforms = v2.Compose([
                ToTensor(),
                Normalize(sample_dir)
                ])
     
    ds, y = get_target_format(args.exp, dates=dates)
    y = np.expand_dims(y, axis= 0)
    
    for i, date in enumerate(dates):
        print(date)
        date_str = date.date().strftime('%Y%m%d')
        sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
        data = dict(np.load(sample), allow_pickle=True)

        x = data['x']

        x, y = transforms((x, y))
        condition = y[0] == 0
        x = torch.unsqueeze(x, dim=0).float()
        y_hat = model(x.to(device)).to(device)
        y_hat = y_hat.detach().cpu()

        unpad_func = UnPad(list(CONFIG[args.exp]['shape']))
        y_hat = unpad_func(y_hat[0])[0].numpy()
        y_hat[condition] = np.nan
        ds.tas[i] = y_hat

    ds.to_netcdf(PREDICTION_DIR/f'tas_day_{data_type}_{period}_r1i1p1f2_gr_{startdate}_{enddate}_{args.exp}_{test_name}.nc')
    