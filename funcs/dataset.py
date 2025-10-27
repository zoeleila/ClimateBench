import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import json

from funcs.hparams import HParams    

class Dataset():
    def __init__(self, data_path, simus):
        self.data_path = data_path
        self.simus = simus

    def load_data(self):
        X_train = []
        Y_train = []

        for i, simu in enumerate(self.simus):

            input_name = 'inputs_' + simu + '.nc'
            output_name = 'outputs_' + simu + '.nc'

            # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
            if 'hist' in simu:
                # load inputs 
                input_xr = xr.open_dataset(self.data_path / input_name)
                    
                # load outputs                                                             
                output_xr = xr.open_dataset(self.data_path / output_name).mean(dim='member')
                output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                            "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                    'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
            
            # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
            else:
                # load inputs 
                input_xr = xr.open_mfdataset([self.data_path / 'inputs_historical.nc', 
                                              self.data_path / input_name]).compute()
                    
                # load outputs                                                             
                output_xr = xr.concat([xr.open_dataset(self.data_path / 'outputs_historical.nc').mean(dim='member'),
                                    xr.open_dataset(self.data_path / output_name).mean(dim='member')],
                                    dim='time').compute()
                output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                            "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                    'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

            print(input_xr.dims, simu)

            # Append to list 
            X_train.append(input_xr)
            Y_train.append(output_xr)
        return X_train, Y_train
    

if __name__ == "__main__":
    
    hparams = HParams()
    dataset = Dataset(hparams.data_path, hparams.simus)

    X_train, Y_train = dataset.load_data()
    meanstd_inputs = {}

    for var in hparams.vars:
        # To not take the historical data into account several time we have to slice the scenario datasets
        # and only keep the historical data once (in the first ssp index 0 in the simus list)
        array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                            [X_train[i][var].sel(time=slice(hparams.len_historical, None)).data for i in range(1, 3)])
        print((array.mean(), array.std()))
        meanstd_inputs[var] = (array.mean(), array.std())
    
    with open(hparams.data_path/'meanstd_dict.json', "w") as f: 
        json.dump(meanstd_inputs, f)