import sys
sys.path.append('.')

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import torch
import glob
from typing import Optional
from torch import Tensor
import xarray as xr

from funcs.hparams import HParams


class ClimateBench(Dataset):
    def __init__(self,
                 transform: Optional[v2.Compose],
                 hparams,
                 data_type: str = 'train'):
        """
        A custom PyTorch Dataset for loading and transforming ClimateBench data.

        Args:
            transform (Optional[v2.Compose]): Transformations to apply to the data.
            hparams : Hyperparameters for the dataset.
            data_type (str): Type of data to load ('train', 'val', or 'test').
        """

        self.transform = transform
        self.data_type = data_type
        self.simus = hparams.simus
        self.slider = hparams.slider
        self.len_historical = hparams.len_historical
        self.var_to_predict = hparams.var_to_predict

        if data_type == 'train' or data_type == 'val':
            self.sample_path = hparams.sample_path / 'train_val'
        else:
            self.sample_path = hparams.sample_path / 'test'
        self.X_train, self.Y_train = self.load_data()

    def input_for_training(self, X_train_xr, slider, skip_historical=False, len_historical=None): 
        X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

        time_length = X_train_np.shape[0]
        # If we skip historical data, the first sequence created has as last element the first scenario data point
        if skip_historical:
            X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(len_historical-slider+1, time_length-slider+1)])
        # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
        else:
            X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(0, time_length-slider+1)])
        return X_train_to_return 
    
    def output_for_training(self, Y_train_xr, var, slider, skip_historical=False, len_historical=None): 
        Y_train_np = Y_train_xr[var].data
        
        time_length = Y_train_np.shape[0]
        
        # If we skip historical data, the first sequence created has as target element the first scenario data point
        if skip_historical:
            Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(len_historical-slider+1, time_length-slider+1)])
        # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
        else:
            Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(0, time_length-slider+1)])
        
        return Y_train_to_return
    
    def load_data(self):
        X_train = []
        Y_train = []

        for i, simu in enumerate(self.simus):

            input_name = 'inputs_' + simu + '.nc'
            output_name = 'outputs_' + simu + '.nc'

            # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
            if 'hist' in simu:
                # load inputs 
                input_xr = xr.open_dataset(self.sample_path / input_name)
                    
                # load outputs                                                             
                output_xr = xr.open_dataset(self.sample_path / output_name).mean(dim='member')
                output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                            "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                    'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
            
            # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
            else:
                # load inputs 
                input_xr = xr.open_mfdataset([self.sample_path / 'inputs_historical.nc', 
                                              self.sample_path / input_name]).compute()
                    
                # load outputs                                                             
                output_xr = xr.concat([xr.open_dataset(self.sample_path / 'outputs_historical.nc').mean(dim='member'),
                                    xr.open_dataset(self.sample_path / output_name).mean(dim='member')],
                                    dim='time').compute()
                output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                            "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                    'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

            print(input_xr.dims, simu)

            # Append to list 
            X_train.append(input_xr)
            Y_train.append(output_xr)
            
        X_train_all = np.concatenate([self.input_for_training(X_train[i], 
                                                            slider=self.slider,
                                                            skip_historical=(i<2), 
                                                            len_historical=self.len_historical) for i in range(len(self.simus))], axis = 0)
        Y_train_all = np.concatenate([self.output_for_training(Y_train[i], 
                                                                self.var_to_predict, 
                                                                skip_historical=(i<2), 
                                                                len_historical=self.len_historical) for i in range(len(self.simus))], axis=0)
        return X_train_all, Y_train_all

    
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.X_train.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: Transformed input (x) and target (y) tensors.
        """
        x = self.X_train[idx]
        y = self.Y_train[idx]
        return x.float(), y.float()
    

if __name__ == "__main__":
    
    hparams = HParams()
    dataset = ClimateBench(transform=None, hparams=hparams, data_type='train')

    print(dataset)