import random
import numpy as np
import torch
import argparse
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from ClimateBench.funcs.dataloader import get_dataloaders
from ClimateBench.funcs.lightning_module import ClimateBenchLightningModule

# Set random seeds for reproducibility
#seed = 6
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)

torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

with open(config_path) as file:
    config = yaml.safe_load(file)

train_dataloader = get_dataloaders('train', config)
val_dataloader = get_dataloaders('val', config)
test_dataloader = get_dataloaders('test', config)

model = ClimateBenchLightningModule(config)

logger = TensorBoardLogger(save_dir=config['model']['runs_dir'], name='lightning_logs')
'''
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", 
    filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min'
)
'''
checkpoint_callback = ModelCheckpoint(monitor=None,
                                      filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}')
torch.set_float32_matmul_precision('high') # For hybrid partition

trainer = pl.Trainer(max_epochs=config['model']['max_epochs'], 
                     default_root_dir=config['model']['runs_dir'],
                     log_every_n_steps=1,
                     accelerator="gpu",
                     devices="auto",
                     precision='16-mixed',
                     logger=logger,
                     callbacks=checkpoint_callback)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')