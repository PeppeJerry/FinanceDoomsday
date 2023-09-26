import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.xavier import CNN_weight_init, FC_weigth_init, LSTM_weigth_init


class CNNBiLSTM(nn.Module):
    def __init__(self, inputDim, out_target, bi_lstm_layers=2, CNN_out=256, dropout=0, dropout_input=0,
                 specific='general', sequence_length=128, SEED=-1, extra="", lr=0.1, lmd=0.001, outLen=14):
        super(CNNBiLSTM, self).__init__()
        if SEED == -1:
            SEED = torch.randint(1, 10000000000000, (1,)).item()
        torch.manual_seed(SEED)
        self.device = torch.device("cpu")

        self.extra = extra
        self.lr = lr
        self.lmd = lmd
        self.specific = specific

        # Defining target time-steps
        self.out_steps = out_target
        self.outLen = outLen

        # CNN Layers to discover patterns in our sequence
        # First CNN Block
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=inputDim, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
        )

        # Weights initialization by considering the number of inputs (Fan-in) and the number of outputs (Fan-out)
        temp_length = sequence_length
        for module in self.cnn1:
            if not (isinstance(module, nn.Conv1d)):
                continue
            temp_length, module = CNN_weight_init(module, temp_length)

        # Second CNN Block (Resizing desired dimensionality)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
        )

        # Weights initialization of cnn2
        for module in self.cnn2:
            if not (isinstance(module, nn.Conv1d)):
                continue
            temp_length, module = CNN_weight_init(module, temp_length)

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=CNN_out, kernel_size=6, padding=1, stride=2),
            nn.ReLU(),
        )

        # Weights initialization of cnn3
        for module in self.cnn3:
            if not (isinstance(module, nn.Conv1d)):
                continue
            temp_length, module = CNN_weight_init(module, temp_length)

        # Layer Normalization 1
        self.layer_norm1 = nn.LayerNorm(normalized_shape=CNN_out)

        # Bi-LSTM to retrive information at different steps in time
        # there are two hidden state arrays equally long CNN_out, one moving forward and the other backward
        self.bi_lstm = nn.LSTM(
            bidirectional=True,  # Bi-LSTM setting
            input_size=CNN_out,
            hidden_size=CNN_out,
            num_layers=bi_lstm_layers,
            batch_first=True
        )
        # Weights initialization of bi_lstm
        self.bi_lstm = LSTM_weigth_init(self.bi_lstm)

        # Layer Normalization 2
        self.layer_norm2 = nn.LayerNorm(normalized_shape=2 * CNN_out)

        # Fully connected
        self.fc = nn.Linear(2 * CNN_out, 4)  # Double because it has forward and backward hidden states
        # Volatility & Price change won't be considered
        self.fc = FC_weigth_init(self.fc, temp_length)  # Weights initialization of fc

        #  Dropout to reduce overfitting
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)

    def forward(self, x):

        # In order to make the algorithm autoregressive, the original sequence x will slide one day at a time
        # The output of each iteration will become the new input until self.outLen outputs are generated
        x_temp = x
        y = torch.zeros(x.size(0), self.outLen, x.size(2)).to(self.device)

        hn = torch.zeros(0, 0, 0).to(self.device)
        cn = torch.zeros(0, 0, 0).to(self.device)
        for i in range(self.outLen):
            if i != 0:
                temp_1 = x_temp[:, 1:, :]
                temp_2 = y[:, :i, :]
                x = torch.cat((temp_1, temp_2), dim=1)

            x = self.dropout_input(x)

            x = x.permute(0, 2, 1)  # (batch_size, steps, channels) -> (batch_size, channels, steps)
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x)

            x = x.permute(0, 2, 1)  # (batch_size, channels, 1) -> (batch_size, 1, channels)
            x = self.layer_norm1(x)

            x = self.dropout(x)

            # Due to CNN layers, the input has dimensionality (Batch_size, 1, channels)
            # In order to efficiently use the recursive architecture, hidden and cell states of previous iterations
            # will be available for the next iteration (i+1)
            if i == 0:
                # If first iteration then they are set by default (Might be improved if hn and cn are trained as well)
                x, (hn, cn) = self.bi_lstm(x)
            else:
                x, (hn, cn) = self.bi_lstm(x, (hn, cn))

            x = self.layer_norm2(x)

            x = self.dropout(x)

            x = self.fc(x)

            y[:, i, :4] = x[:, 0, :4]
        return y

    def train_model(self, train_loader, x_valid, Y_valid, epochs=2000,
                    path="models/CNN_BiLSTM_Weights/", extra="", custom_loss=False, threshold=100):
        # Defining if CUDA is available for processing, use CPU otherwise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.device = device
        lr = self.lr
        # If "cuda" is available data will move accordingly
        x_valid = x_valid.to(device)
        Y_valid = Y_valid.to(device)

        # AdamW optimizer
        # Better L2 Regularization compared to Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.lmd)

        # MSE loss
        criterion = nn.MSELoss()

        # Start training
        self.train()

        # Best losses encountered during training
        loss_counter = float("inf")
        loss_val_counter = float("inf")

        # Loss history
        losses = []
        losses_val = []
        batch_iter = iter(train_loader)

        LowLR = 0
        for epoch in range(epochs):
            LowLR = LowLR + 1
            optimizer.zero_grad()

            # Checking if iterator has a batch for the iteration
            try:
                x_batch, Y_batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                x_batch, Y_batch = next(batch_iter)

            # If "cuda" is available data will move accordingly
            x_batch = x_batch.to(device)
            Y_batch = Y_batch.to(device)

            output = self(x_batch)

            if custom_loss:
                # Personalized loss function

                # 1) Loss will be evaluated for indexes self.out_steps
                # if self.out_steps = [0,6,13] is the same as increasing predict accuracies on 1 day, 1 week, 2 weeks

                # 2) Loss will not consider all variables since we are interested in (Open, Close, High, Low)
                # These variables will always be placed in the first 4 places so it possible to exploit this information
                # NOTE: the model is still taking variables such as "Volume" as input
                loss = criterion(output[:, self.out_steps, :4], Y_batch[:, self.out_steps, :4])
            else:
                # Standard loss by considering just the first 4 features (Open, Close, High, Low) for better accuracy
                loss = criterion(output[:, :, :4], Y_batch[:, :, :4])

            losses.append(loss.item())
            if loss.item() < loss_counter:
                loss_counter = loss.item()
                print("Training [{},{}]".format(epoch + 1, loss_counter))

            # The overall training process will not consider these lines below
            with torch.no_grad():
                output = self(x_valid)
                loss_val = criterion(output[:, :, :4], Y_valid[:, :, :4])
                # loss_val = criterion(output[:, self.out_steps, :4], Y_valid[:, self.out_steps, :4])
                losses_val.append(loss_val.item())

                # Saving best parameters based on validation set
                if loss_val.item() < loss_val_counter:
                    LowLR = 0
                    threshold = max(100, int(threshold * 0.05))
                    loss_val_counter = loss_val.item()
                    print("Validation [{},{}]".format(epoch + 1, loss_val_counter))
                    torch.save(self.state_dict(), path + self.specific + extra + '.pth')

            loss.backward()
            optimizer.step()

            if LowLR >= threshold:
                lr = lr / 2
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.lmd)
                LowLR = 0

        # Taking the best parameters obtained for the validation set
        self.load_state_dict(torch.load(path + self.specific + extra + '.pth'))
        self.eval()
        self.switch_CPU()
        return losses, losses_val

    def validation_loss(self, x, Y):
        with torch.no_grad():
            output = self(x)
            criterion = nn.MSELoss()
            loss = criterion(output, Y)
            return loss.item()

    def switch_CPU(self):
        device = torch.device("cpu")
        self.to(device)
        self.device = device
        return True

    def switch_CUDA(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.device = device
        return torch.cuda.is_available()
