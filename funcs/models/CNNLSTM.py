import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

import random 
seed = 6 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



class CNNLSTMModel(nn.Module):
    def __init__(self, slider, height=96, width=144, channels=4,
                 conv_filters=20, conv_kernel=(3, 3), pool_size=2, lstm_units=25):
        super(CNNLSTMModel, self).__init__()
        self.slider = slider
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.pool_size = pool_size
        self.lstm_units = lstm_units

        # CNN layers
        self.conv = nn.Conv2d(channels, conv_filters, 
                              kernel_size=conv_kernel, 
                              padding='same')
        self.pool = nn.AvgPool2d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # LSTM input = number of filters after CNN pooling
        self.lstm_input_size = conv_filters
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                            hidden_size=lstm_units, 
                            batch_first=True)
        self.fc = nn.Linear(lstm_units, height * width)

    def forward(self, x):
        """
        x: tensor of shape (batch, time, height, width, channels)
        """
        batch_size, time_steps, height, width, channels = x.size()

        cnn_features = []
        for t in range(time_steps):
            # Rearrange (batch, height, width, channels) -> (batch, channels, height, width)
            frame = x[:, t].permute(0, 3, 1, 2)
            out = F.relu(self.conv(frame))
            out = self.pool(out)
            out = self.global_avg_pool(out)
            out = out.view(batch_size, -1)  # (batch, conv_filters)
            cnn_features.append(out)

        # Stack over time → (batch, time, conv_filters)
        cnn_features = torch.stack(cnn_features, dim=1)

        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = F.relu(lstm_out[:, -1, :])  # last timestep

        # Dense and reshape to image
        out = self.fc(lstm_out)
        out = out.view(batch_size, 1, self.height, self.width)
        return out

def get_CNNLSTM_paper(in_shape, out_shape):
    keras.backend.clear_session()
    cnn_model = None

    cnn_model = Sequential()
    cnn_model.add(Input(shape=in_shape))
    cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'))) # , input_shape=in_shape))
    cnn_model.add(TimeDistributed(AveragePooling2D(2)))
    cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
    cnn_model.add(LSTM(25, activation='relu'))
    cnn_model.add(Dense(np.prod(out_shape)))
    cnn_model.add(Activation('linear'))
    cnn_model.add(Reshape(out_shape))
    return cnn_model

# Example usage
if __name__ == "__main__":
    #model = CNNLSTMModel(slider=5)
    x = torch.randn(2, 5, 96, 144, 4)  # (batch, time, height, width, channels)
    in_shape = x.shape[1:]
    out_shape = x.shape[1:]
    model = get_CNNLSTM_paper(in_shape, out_shape)
    y = model(x)
    print(y.shape)  # Expected: (2, 1, 96, 144)
