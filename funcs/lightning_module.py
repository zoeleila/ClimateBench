"""

"""

import sys
sys.path.append('.')

from pathlib import Path
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError

from ClimateBench.funcs.models.CNNLSTM import CNNLSTMModel

layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
    },
}

class ClimateBenchLightningModule(pl.LightningModule):
    def __init__(self, config:dict):
        super().__init__()
        self.slider = config['data']['slider']
        self.learning_rate = config['model']['learning_rate']
        self.runs_dir = config['model']['runs_dir']
        self.var_to_predict = config['model']['var_to_predict']
        self.vars = config['data']['vars']
        self.img_size = config['model']['img_size']
        self.scheduler_step_size = config['model']['scheduler_step_size']
        self.scheduler_gamma = config['model']['scheduler_gamma']
        os.makedirs(self.runs_dir, exist_ok=True)

        self.loss = nn.MSELoss()
        
        self.metrics_dict = nn.ModuleDict({
                    "rmse": MeanSquaredError(squared=True),
                    "mae": MeanAbsoluteError()
                })
        self.spatial_corr_metric = PearsonCorrCoef()

        self.model = CNNLSTMModel(self.slider, height=96, width=144, channels=4).float()

        self.test_metrics = {}
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs_true = []
        self.test_step_outputs_hat = []
        
        self.save_hyperparameters()
        self.epoch_start_time = None

    def forward(self, x):
        return self.model(x) 

    def on_train_start(self):
        self.logger.experiment.add_custom_scalars(layout)
        self.logger.log_hyperparams(vars(self.hparams))

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def common_step(self, x, y):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.common_step(x, y)
        self.train_step_outputs.append(loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.train_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/train", epoch_average, self.current_epoch)
        self.train_step_outputs.clear()
        epoch_duration = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_duration, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.common_step(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.val_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/val", epoch_average, self.current_epoch)
        self.val_step_outputs.clear()

        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.common_step(x, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics_dict.items():
            metric.update(y_hat, y)
            batch_dict[metric_name] = metric.compute()
            self.logger.experiment.add_scalar(metric_name, metric.compute(), batch_idx)
            metric.reset()
        self.test_metrics[batch_idx] = batch_dict

        y_flat = y.flatten()
        y_hat_flat = y_hat.flatten()
        self.spatial_corr_metric.update(y_hat_flat, y_flat)

        self.test_step_outputs_true.append(y.cpu().numpy().mean())
        self.test_step_outputs_hat.append(y_hat.cpu().numpy().mean())

        if batch_idx == 0:

            fig, ax = plt.subplots()
            vmin, vmax = np.min(y.cpu().numpy()), np.max(y.cpu().numpy())
            print(vmin, vmax)
            levels = np.linspace(vmin, vmax, 11)
            cs = ax.contourf(y[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_y_0', fig) 
    
            fig, ax = plt.subplots()
            cs = ax.contourf(y_hat[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_yhat_0', fig)
 
            
    def build_metrics_dataframe(self):
        data = []
        first_sample = list(self.test_metrics.keys())[0]
        metrics = list(self.test_metrics[first_sample].keys())
        for name_sample, metrics_dict in self.test_metrics.items():
            data.append([name_sample] + [metrics_dict[m].item() for m in metrics])
        return pd.DataFrame(data, columns=["Name"] + metrics)

    def save_test_metrics_as_csv(self, df):
        path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
        df.to_csv(path_csv, index=False)
    
    def on_test_epoch_end(self):
        df = self.build_metrics_dataframe()
        self.save_test_metrics_as_csv(df)
        df = df.drop("Name", axis=1)
        self.log('hp_metric', df['rmse'].mean())
        self.log('loss', df['loss'].mean())

        spatial_corr = self.spatial_corr_metric.compute()
        self.log("hp_metric_corr", spatial_corr)
        
        fig, ax = plt.subplots()
        ax.plot(self.test_step_outputs_true, label='True')
        ax.plot(self.test_step_outputs_hat, label='Predicted')
        ax.set_xlabel('year')
        ax.set_ylabel(f'{self.var_to_predict} value')
        ax.legend()
        self.logger.experiment.add_figure('Figure/test_true_vs_predicted', fig)


    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]

