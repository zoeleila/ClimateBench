"""
Compute scores and evaluation plots on test set using a trained model.
"""


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

parser = argparse.ArgumentParser(description="Predict and plot results for full period")
parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint')
args = parser.parse_args()

hparams_file = Path(args.checkpoint).parent.parent / 'hparams.yaml'
device = 'cpu'
model = ClimateBenchLightningModule.load_from_checkpoint(args.checkpoint, map_location=device, hparams_file = hparams_file)
model.eval()
config = model.hparams['config']
test_dataloader = get_dataloaders('test', config)

rmse = MeanSquaredError()
mae = MeanAbsoluteError()
pcc = PearsonCorrCoef()
var_to_predict = config['model']['var_to_predict']
test_name = config['model']['test_name']
exp = config['data']['exp']
plots = EvaluationPlots(simulation_name=config['model']['simus_test'][0],
                        var_name=var_to_predict)
graph_path = GRAPHS_DIR / exp / test_name
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
                        save_path = graph_path / f'{test_name}_time_series_plot.png')
plots.plot_spatial_map(y_full.numpy().squeeze(), 
                        y_hat_full.numpy().squeeze(), 
                        time_index=None,
                        save_path = graph_path/  f'{test_name}_spatial_map_plot.png')
plots.plot_error_maps(y_full.numpy().squeeze(), 
                        y_hat_full.numpy().squeeze(),
                        save_path = graph_path / f'{test_name}_error_maps.png')