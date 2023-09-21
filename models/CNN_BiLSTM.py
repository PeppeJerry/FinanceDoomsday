import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def denormalize(x_norm, norm="minmax", min=6, max=9, mean=4, std=2):
    if norm == "minmax":
        x_pred = x_norm * (max - min) + min
        return x_pred


class CNNBiLSTM(nn.Module):
    def __init__(self, inputDim, out_target, bi_lstm_layers=2, CNN_out=256, dropout=0.5, dropout_input=0,
                 specific='general', sequence_length=128, SEED=-1):
        super(CNNBiLSTM, self).__init__()
        if SEED == -1:
            SEED = torch.randint(1, 10000000000000, (1,)).item()
        torch.manual_seed(SEED)
        self.specific = specific

        # Defining target time-steps
        self.out_steps = out_target

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
            temp_length = self.CNN_weight_init(module, temp_length)

        # Second CNN Block (Resizing desired dimensionality)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=CNN_out, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
        )

        # Weights initialization of cnn2
        for module in self.cnn2:
            if not (isinstance(module, nn.Conv1d)):
                continue
            temp_length = self.CNN_weight_init(module, temp_length)

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
        self.LSTM_weigth_init(self.bi_lstm)

        # Layer Normalization 2
        self.layer_norm2 = nn.LayerNorm(normalized_shape=2 * CNN_out)

        # Fully connected
        self.fc = nn.Linear(2 * CNN_out, 4)  # Double because it has forward and backward hidden states
        # Volatility & Price change won't be considered
        self.FC_weigth_init(self.fc, temp_length)  # Weights initialization of fc

        #  Dropout to reduce overfitting
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.permute(0, 2, 1)  # (batch_size, steps, channels) -> (batch_size, channels, steps)
        x = self.cnn1(x)

        x = self.cnn2(x)

        x = x.permute(0, 2, 1)  # (batch_size, channels, steps) -> (batch_size, steps, channels)
        x = self.layer_norm1(x)

        x = self.dropout(x)

        x, _ = self.bi_lstm(x)

        x = self.layer_norm2(x)

        x = self.dropout(x)

        x = self.fc(x)
        return x

    def set_company(self, company):
        self.specific = company

    def train_model(self, train_loader, x_valid, Y_valid, epochs=1000, lr=0.001, lambd=0.02,
                    path="models/CNN_BiLSTM_Weights/", extra=""):
        # Defining if CUDA is available for processing, use CPU otherwise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # If "cuda" is available data will move accordingly
        x_valid = x_valid.to(device)
        Y_valid = Y_valid.to(device)

        # AdamW optimizer
        # Better L2 Regularization compared to Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lambd)

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

        for epoch in range(epochs):
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

            # Personalized loss function

            # 1) Loss will be evaluated for indexes self.out_steps
            # if self.out_steps = [0,6,13] is the same as increasing predict accuracies on 1 day, 1 week, 2 weeks

            # 2) Loss will not consider all variables since we are interested in (Open, Close, High, Low)
            # These variables will always be placed in the first 4 places so it possible to exploit this information
            # NOTE: the model is still taking variables such as "Volume" as input
            loss = criterion(output[:, self.out_steps, :4], Y_batch[:, self.out_steps, :4])
            #loss = criterion(output[:, :, :4], Y_batch[:, :, :4])

            losses.append(loss.item())
            if loss.item() < loss_counter:
                loss_counter = loss.item()
                print("Training [{},{}]".format(epoch + 1, loss_counter))

            # The overall training process will not consider these lines below
            with torch.no_grad():
                output = self(x_valid)
                loss_val = criterion(output[:, self.out_steps, :4], Y_valid[:, self.out_steps, :4])
                losses_val.append(loss_val.item())

                # Saving best parameters based on validation set
                if loss_val.item() < loss_val_counter:
                    loss_val_counter = loss_val.item()
                    print("Validation [{},{}]".format(epoch + 1, loss_val_counter))
                    torch.save(self.state_dict(), path + self.specific + extra + '.pth')

            loss.backward()
            optimizer.step()

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

    def denormalize(self, x, min_vect=6, max_vect=9, mean=4, std=2):
        y_norm = self(x)
        return denormalize(y_norm, min=min_vect, max=max_vect)

    def switch_CPU(self):
        device = torch.device("cpu")
        self.to(device)
        return True

    def switch_CUDA(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        return torch.cuda.is_available()

    def CNN_weight_init(self, module, sequence_length):

        # Number of filters
        out_dim = module.out_channels

        # Input channels
        inputDim = module.in_channels

        # Config parameters of nn.Conv1d() module
        F = module.kernel_size[0]
        P = module.padding[0]
        S = module.stride[0]

        # Evaluating fan-in (n_in) and fan-out (n_out)
        n_in = sequence_length * inputDim
        temp_length = int((sequence_length + 2 * P - F) / S + 1)  # Feature map dimension
        n_out = int(temp_length * out_dim)

        # Applying Xavier (Glorot) initialization
        weight_coefficient = torch.tensor(2 / (n_in + n_out))
        weight_coefficient = torch.sqrt(weight_coefficient).item()
        module.weight.data = torch.randn(module.weight.data.size()) * weight_coefficient
        module.bias.data = torch.randn(module.bias.data.size()) * weight_coefficient

        # Returning the new sequence length after applying convolution
        return temp_length

    def FC_weigth_init(self, module, sequence_length):
        # Evaluating fan-in (n_in) and fan-out (n_out)
        n_in = sequence_length * module.in_features
        n_out = sequence_length * module.out_features

        # Applying Xavier (Glorot) initialization
        weight_coefficient = torch.tensor(2 / (n_in + n_out))
        weight_coefficient = torch.sqrt(weight_coefficient).item()
        module.weight.data = torch.randn(module.weight.data.size()) * weight_coefficient
        module.bias.data = torch.randn(module.bias.data.size()) * weight_coefficient

    def LSTM_weigth_init(self, module):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
