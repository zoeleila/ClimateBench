
import torch
import argparse
import numpy as np
from torchvision.transforms import v2
import yaml
from pathlib import Path
from tqdm import tqdm
from torchmetrics import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef

from ClimateBench.funcs.lightning_module import ClimateBenchLightningModule
from ClimateBench.funcs.dataloader import get_dataloaders
from utils.plotutils import EvaluationPlots
from ClimateBench.funcs.settings import GRAPHS_DIR

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


def get_target_format(config):
    simus_test = config['model']['simus_test']
    var_to_predict = config['model']['var_to_predict']
    ds_target = xr.open_dataset(config['data']['raw_dir']/f'output_{simus_test[0]}.nc')
    ds = xr.Dataset(
        data_vars={var_to_predict: (['time', 'lat', 'lon'], np
'''



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

    rmse = MeanSquaredError()
    mae = MeanAbsoluteError()
    pcc = PearsonCorrCoef()
    var_to_predict = config['model']['var_to_predict']
    exp = config['model']['exp']
    plots = EvaluationPlots(simulation_name=config['model']['simus_test'][0],
                            var_name=var_to_predict)
    graph_path = GRAPHS_DIR / exp
    graph_path.mkdir(parents=True, exist_ok=True)

    y_full = []
    y_hat_full = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        x, y = batch
        x = x.float().to(device)
        with torch.no_grad():
            y_hat = model(x).cpu()   
        rmse.update(y_hat, y)
        mae.update(y_hat, y)
        pcc.update(y_hat.flatten(), y.flatten()) # spatial correlation
        y_full.append(y)
        y_hat_full.append(y_hat)
    y_full = torch.cat(y_full, dim=0)
    y_hat_full = torch.cat(y_hat_full, dim=0)

    rmse_val = rmse.compute().item()**0.5
    mae_val = mae.compute().item()
    pcc_val = pcc.compute().item()
    print(f'RMSE: {rmse_val}, MAE: {mae_val}, PCC: {pcc_val}')

    plots.plot_time_series(y_full.numpy().squeeze(), 
                           y_hat_full.numpy().squeeze(),
                           save_path = graph_path / f'{var_to_predict}_time_series_plot.png')
    plots.plot_spatial_map(y_full.numpy().squeeze(), 
                           y_hat_full.numpy().squeeze(), 
                           time_index=None,
                           save_path = graph_path/  f'{var_to_predict}_spatial_map_plot.png')
    plots.plot_error_maps(y_full.numpy().squeeze(), 
                          y_hat_full.numpy().squeeze(),
                          save_path = graph_path / f'{var_to_predict}_error_maps.png')