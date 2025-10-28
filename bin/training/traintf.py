import sys
sys.path.append('.')

import json
import numpy as np

from funcs.hparams import HParams
from funcs.dataset import Dataset


class Transforms():
    def __init__(self, data_path):
        self.meanstd_dict = json.load(open(data_path / 'meanstd_dict.json'))

    def normalize(self, data, var):
        mean = self.meanstd_dict[var][0]
        std = self.meanstd_dict[var][1]
        return (data - mean)/std

    def unnormalize(self, data, var):
        mean = self.meanstd_dict[var][0]
        std = self.meanstd_dict[var][1]
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

    hparams = HParams()
    transforms = Transforms(data_path=hparams.data_path)
    dataset = Dataset(data_path=hparams.data_path, simus=hparams.simus)

    X_train, Y_train = dataset.load_data()

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2
    import random 
    from tensorboard import SummaryWriter
    seed = 6 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train_norm = [] 
    for i, train_xr in enumerate(X_train): 
        for var in hparams.vars: 
            var_dims = train_xr[var].dims
            train_xr=train_xr.assign({var: (var_dims, transforms.normalize(train_xr[var].data, var))}) 
        X_train_norm.append(train_xr)
   
    for var_to_predict in hparams.vars_to_predict:
        
        print(var_to_predict)
        
        # Data
        X_train_all = np.concatenate([transforms.input_for_training(X_train_norm[i], 
                                                                    hparams.slider, 
                                                                    skip_historical=(i<2), 
                                                                    len_historical=hparams.len_historical) for i in range(len(hparams.simus))], axis = 0)
        Y_train_all = np.concatenate([transforms.output_for_training(Y_train[i], 
                                                                     var_to_predict, 
                                                                     hparams.slider, 
                                                                     skip_historical=(i<2), 
                                                                     len_historical=hparams.len_historical) for i in range(len(hparams.simus))], axis=0)
        print(X_train_all.shape)
        print(Y_train_all.shape)
        
        # Model    
        keras.backend.clear_session()
        cnn_model = CNNModel(slider=hparams.slider).build()
        cnn_model.summary()
        cnn_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=hparams.learning_rate), 
                  loss=keras.losses.MeanSquaredError(), 
                  metrics=["mse"])

        run_dir = hparams.runs_path / hparams.exp / var_to_predict
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_filepath = run_dir / 'best-checkpoint-{val_loss:.2f}.keras'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            )
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=run_dir/ 'logs', histogram_freq=1)

        # add custom scalars layout for TensorBoard (uses tensorboardX if available)
        try:
            layout = {
            "Check Overfit": {
                "loss": ["Multiline", ["train_loss", "val_loss"]],
            },
            }
            writer = SummaryWriter(log_dir=str(run_dir / 'logs'))
            writer.add_custom_scalars(layout)
            writer.close()
        except Exception:
            # tensorboardX not installed or failed -> skip custom layout
            pass

        hist = cnn_model.fit(X_train_all,
                    Y_train_all,
                    batch_size=hparams.batch_size, 
                    epochs=hparams.epoch,
                    verbose=1,
                    callbacks=[model_checkpoint_callback, tb_callback])
        hist.history