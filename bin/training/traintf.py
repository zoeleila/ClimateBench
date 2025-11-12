import json
import numpy as np
import xarray as xr
import argparse
import yaml
from pathlib import Path

from ClimateBench.funcs.settings import PREDICTIONS_DIR

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

            print(simu)
            print(output_xr)
            # Append to list 
            X_train.append(input_xr)
            Y_train.append(output_xr)
        return X_train, Y_train

class Transforms():
    def __init__(self):
        pass

    def normalize(self, data, var, meanstd_dict):
        mean = meanstd_dict[var][0]
        std = meanstd_dict[var][1]
        return (data - mean)/std

    def unnormalize(self, data, var, meanstd_dict):
        mean = meanstd_dict[var][0]
        std = meanstd_dict[var][1]
        return data * std + mean

    # Functions for reshaping the data 
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
    
    
class CNNModel:
    def __init__(self, slider, height=96, width=144, channels=4,
                    conv_filters=20, conv_kernel=(3, 3), pool_size=2, lstm_units=25):
        self.slider = slider
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.pool_size = pool_size
        self.lstm_units = lstm_units

    def build(self):
        model = Sequential()
        model.add(Input(shape=(self.slider, self.height, self.width, self.channels)))
        model.add(TimeDistributed(Conv2D(self.conv_filters, self.conv_kernel, padding='same', activation='relu')))
        model.add(TimeDistributed(AveragePooling2D(self.pool_size)))
        model.add(TimeDistributed(GlobalAveragePooling2D()))
        model.add(LSTM(self.lstm_units, activation='relu'))
        model.add(Dense(1 * self.height * self.width))
        model.add(Activation('linear'))
        model.add(Reshape((1, self.height, self.width)))
        return model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.safe_load(file)

    transforms = Transforms()
    dataset = Dataset(data_path=Path(config['data']['raw_dir']), simus=config['model']['simus_train'])
    len_historical = config['data']['len_historical']
    var_to_predict = config['model']['var_to_predict']
    slider = config['data']['slider']
    simus_train = config['model']['simus_train']
    simus_train.remove('historical')
    exp = config['data']['exp']
    predictions_dir = PREDICTIONS_DIR / exp

    X_train, Y_train = dataset.load_data()
    meanstd_inputs = {}

    for var in config['data']['vars']:
        # To not take the historical data into account several time we have to slice the scenario datasets
        # and only keep the historical data once (in the first ssp index 0 in the simus list)
        array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                            [X_train[i][var].sel(time=slice(len_historical, None)).data for i in range(1, 3)])
        print((array.mean(), array.std()))
        meanstd_inputs[var] = (array.mean(), array.std())
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2
    print(tf.config.list_physical_devices('GPU'))
    import random 
    seed = 6 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train_norm = [] 
    for i, train_xr in enumerate(X_train): 
        for var in config['data']['vars']: 
            var_dims = train_xr[var].dims
            train_xr=train_xr.assign({var: (var_dims, transforms.normalize(train_xr[var].data, var, meanstd_inputs))}) 
        X_train_norm.append(train_xr)

    
    print(var_to_predict)
    
    # Data
    X_train_all = np.concatenate([transforms.input_for_training(X_train_norm[i], 
                                                                slider, 
                                                                skip_historical=(i<2), 
                                                                len_historical=len_historical) for i in range(len(simus_train))], axis = 0)
    Y_train_all = np.concatenate([transforms.output_for_training(Y_train[i], 
                                                                    var_to_predict, 
                                                                    slider, 
                                                                    skip_historical=(i<2), 
                                                                    len_historical=len_historical) for i in range(len(simus_train))], axis=0)
    print(X_train_all.shape)
    print(Y_train_all.shape)

   
    # Model    
    keras.backend.clear_session()
    cnn_model = CNNModel(slider=slider).build()
    cnn_model.summary()
    cnn_model.compile(optimizer=keras.optimizers.RMSprop(), 
                loss=keras.losses.MeanSquaredError(), 
                metrics=["mse"])

    hist = cnn_model.fit(X_train_all,
                    Y_train_all,
                    use_multiprocessing=True, 
                    #workers=5,
                    batch_size=16,
                    epochs=30,
                    verbose=1,
                    validation_split=0.2)
    hist.history

    simu_test = config['model']['simus_test'][0]
    X_test = xr.open_mfdataset([Path(config['data']['raw_dir']) / 'inputs_historical.nc',
                            Path(config['data']['raw_dir']) / f'inputs_{simu_test}.nc']).compute()

    # Normalize input data 
    for var in config['data']['vars']: 
        var_dims = X_test[var].dims
        X_test = X_test.assign({var: (var_dims, transforms.normalize(X_test[var].data, var, meanstd_inputs))}) 
        
    X_test_np = transforms.input_for_training(X_test, skip_historical=False, len_historical=len_historical) 

    m_pred = cnn_model.predict(X_test_np)
    # Reshape to xarray 
    m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
    m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[X_test.time.data[slider-1:], X_test.latitude.data, X_test.longitude.data])
    xr_prediction = m_pred.transpose('lat', 'lon', 'time').sel(time=slice(2015,2101)).to_dataset(name=var_to_predict)

    if var_to_predict=="pr90" or var_to_predict=="pr":
        xr_prediction = xr_prediction.assign({var_to_predict: xr_prediction[var_to_predict] / 86400})

    # Save test predictions as .nc 
    xr_prediction.to_netcdf(predictions_dir / f'{simu_test}_2015_2100_paper_{var_to_predict}.nc')
    xr_prediction.close()
     