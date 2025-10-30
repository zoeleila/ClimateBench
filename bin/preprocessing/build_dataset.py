'''
Create samples for training using ClimateBench raw netCDF files.

Duncan Watson-Parris, Shahine, & Matthew Chantry. (2022). 
duncanwp/ClimateBench: v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.7064303

Modified by: Zoé Garcia 
Date : October 2025

'''

import yaml
import xarray as xr
from pathlib import Path
import argparse
import numpy as np

class BuilderDataset:
    def __init__(self,
                 raw_dir:Path,
                 dataset_dir:Path,
                 slider:int,
                 len_historical:int):
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir
        self.slider = slider
        self.len_historical = len_historical

    def load_data(self, simu):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'

        # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
        if 'hist' in simu:
            # load inputs 
            input_xr = xr.open_dataset(self.raw_dir / input_name)
                
            # load outputs                                                             
            output_xr = xr.open_dataset(self.raw_dir / output_name).mean(dim='member')
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                        "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop_vars(['quantile'])
        
        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs 
            input_xr = xr.open_mfdataset([self.raw_dir / 'inputs_historical.nc', 
                                            self.raw_dir / input_name]).compute()
                
            # load outputs                                                             
            output_xr = xr.concat([xr.open_dataset(self.raw_dir / 'outputs_historical.nc').mean(dim='member'),
                                xr.open_dataset(self.raw_dir / output_name).mean(dim='member')],
                                dim='time').compute()
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                        "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop_vars(['quantile'])
        if input_xr.time.shape[0] != output_xr.time.shape[0]:
            last_year = min(input_xr.time.data[-1], output_xr.time.data[-1])
            input_xr = input_xr.sel(time=slice(None, last_year))
            output_xr = output_xr.sel(time=slice(None, last_year))
        input_order = ['CO2', 'SO2', 'CH4', 'BC']
        input_xr = input_xr[input_order]
        output_order = ['diurnal_temperature_range', 'tas', 'pr', 'pr90']
        output_xr = output_xr[output_order]
        return input_xr, output_xr
    
    def time_formating(self, data, simu, input=False):
        time_length = data.shape[0]
        if input :
            if 'ssp'in simu:
                array = np.array([data[i:i+self.slider] for i in range(self.len_historical - self.slider + 1, time_length - self.slider + 1)])
            else:
                array = np.array([data[i:i+self.slider] for i in range(0, time_length - self.slider + 1)])
        else:
            if 'ssp'in simu:
                array = np.array([[data[i + self.slider - 1]] for i in range(self.len_historical - self.slider + 1, time_length - self.slider + 1)])
            else:
                array = np.array([[data[i + self.slider - 1]] for i in range(0, time_length - self.slider + 1)])
        return array
    

    def build_samples(self, simu):
        X_xr, Y_xr = self.load_data(simu)
        X_np = self.time_formating(X_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data,
                                   simu=simu,
                                   input=True)
        Y_np = self.time_formating(Y_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data,
                                   simu=simu)
        time = self.time_formating(Y_xr.time.data,
                                   simu=simu)
        return X_np, Y_np, time
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Experiment name corresponding to a config file')
    args = parser.parse_args()
    config = args.config

    with open(config) as file:
        config = yaml.safe_load(file)

    simus = config['data']['simus']
    slider = config['data']['slider']
    len_historical = config['data']['len_historical']
    raw_dir = Path(config['data']['raw_dir'])
    dataset_dir = Path(config['data']['dataset_dir'])

    builder = BuilderDataset(raw_dir=raw_dir,
                             dataset_dir=dataset_dir,
                             slider=slider,
                             len_historical=len_historical)
    
    for i, simu in enumerate(simus):
        print(simu)
        X_np, Y_np, time = builder.build_samples(simu)
        for i, year in enumerate(np.squeeze(time)):
            print(year)
            sample = {'x': X_np[i],
                      'y': Y_np[i]}
            sample_name = f'sample_{simu}_{year}.npz'
            np.savez(dataset_dir / sample_name, **sample)
            
        
