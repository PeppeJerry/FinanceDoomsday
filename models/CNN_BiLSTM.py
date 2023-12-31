import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.xavier import CNN_weight_init, FC_weight_init, LSTM_weight_init


class CNNBiLSTM(nn.Module):
    def __init__(self, inputDim, out_target=None, bi_lstm_layers=2, CNN_out=32, dropout=0, dropout_input=0, kernel=5,
                 specific='general', sequence_length=32, SEED=-1, extra="", lr=0.1, lmd=0.001, out_len=7, outputDim=4,
                 noise=None, auto_regressive=None, path=""):
        super(CNNBiLSTM, self).__init__()
        if SEED == -1:
            SEED = torch.randint(1, 10000000000000, (1,)).item()
        torch.manual_seed(SEED)

        ######################
        # Model's parameters #
        ######################

        self.lr = lr
        self.lmd = lmd
        self.drop_prop = dropout
        self.drop_propIN = dropout_input

        self.inputDim = inputDim
        self.outputDim = outputDim

        self.path = path
        self.specific = specific
        self.extra = extra

        # Defining target time-steps
        if out_target is None:
            self.out_steps = [0, 1, 6]
        else:
            self.out_steps = out_target
        self.out_len = out_len

        #############################
        # Convolutional layer Stack #
        #############################

        self.layer_norm0 = nn.LayerNorm(normalized_shape=inputDim)

        padding = int((kernel - 1) / 2)

        # First CNN Block
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=inputDim, out_channels=8, kernel_size=kernel, padding=padding, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=kernel, padding=padding, stride=1),
            nn.ReLU(),
        )
        # Second CNN Block
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=kernel, padding=padding, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=20, kernel_size=kernel, padding=padding, stride=1),
            nn.ReLU(),
        )
        # Third CNN Block
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=24, kernel_size=kernel, padding=padding, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=24, out_channels=CNN_out, kernel_size=kernel, padding=padding, stride=1),
            nn.ReLU(),
        )

        # Layer Normalization 1
        self.layer_norm1 = nn.LayerNorm(normalized_shape=CNN_out)

        ################
        # BiLSTM layer #
        ################

        BiLSTM = True
        # Bi-LSTM
        # there are two hidden state arrays equally long CNN_out, one moving forward and the other backward
        self.bi_lstm = nn.LSTM(
            bidirectional=True,  # Bi-LSTM setting
            input_size=CNN_out,
            hidden_size=CNN_out,
            num_layers=bi_lstm_layers,
            batch_first=True
        )

        LSTM_shape = 2 * CNN_out if BiLSTM else CNN_out

        ################
        # Output layer #
        ################

        # Fully connected
        self.fc = nn.Linear(LSTM_shape, outputDim)

        #########################
        # Xavier initialization #
        #########################

        # Weights initialization
        temp_length = sequence_length
        for CNN in [self.cnn1, self.cnn2, self.cnn3]:
            for module in CNN:
                if not (isinstance(module, nn.Conv1d)):
                    continue
                temp_length, module = CNN_weight_init(module, temp_length)

        # Weights initialization of bi_lstm
        self.bi_lstm = LSTM_weight_init(self.bi_lstm)

        # Weights initialization of fc
        self.fc = FC_weight_init(self.fc, temp_length)

        #  Dropout to reduce overfitting for neurons and input
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)

    def forward(self, x, Y=None, training=None):

        batch = x.size(0)

        # Input
        # x = self.layer_norm0(x)
        x = self.dropout_input(x)

        # Convolutional stack
        x = x.permute(0, 2, 1)  # (batch_size, steps, channels) -> (batch_size, channels, steps)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.permute(0, 2, 1)  # (batch_size, channels, steps) -> (batch_size, steps, channels)

        # Saving it to decrease the number of operations during the autoregressive cycle
        cnn_x = x.to(x.device)

        # Layer normalization & Dropout
        x = self.layer_norm1(x)
        x = self.dropout(x)

        x, _ = self.bi_lstm(x)

        # Output batch with dimensionality (batch, out_len, outputDim)
        Y = torch.zeros((batch, self.out_len, self.outputDim)).to(x.device)
        Y[:, 0, :] = self.fc(x[:, [-1], :]).squeeze()

        # Autoregressive algorithm, the last output (Y[:, t, :])  will be given as input at (t+1) iteration
        # NOTE: inputDim is higher from OutputDim so remaining columns are filled with zeros
        last = torch.zeros((batch, 1, self.inputDim)).to(x.device)
        last[:, 0, :4] = Y[:, 0, :]

        for t in range(1, self.out_len):
            # It is not needed to perform convolution on every sample but just to the last one
            last = last.permute(0, 2, 1)
            last = self.cnn1(last)
            last = self.cnn2(last)
            last = self.cnn3(last)
            last = last.permute(0, 2, 1)

            # Concatenating previous convoluted data with the last one
            cnn_x = torch.cat((cnn_x, last), dim=1)
            x = self.layer_norm1(cnn_x)

            # To effectively make use of the bidirectional recurrent layer,
            # past CNN outputs will be passed with the new sample "last" to generate the new output
            x, _ = self.bi_lstm(x)

            Y[:, t, :] = self.fc(x[:, [-1], :]).squeeze()
            last = torch.zeros((batch, 1, self.inputDim)).to(x.device)
            last[:, 0, :4] = Y[:, t, :]

        return Y

    def validation_loss(self, x, Y):
        with torch.no_grad():
            output = self(x)
            criterion = nn.MSELoss()
            loss = criterion(output, Y)
            return loss.item()
