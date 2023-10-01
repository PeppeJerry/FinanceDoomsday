import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.xavier import CNN_weight_init, FC_weight_init, LSTM_weight_init, enc_dec_weight_init


class Parallel_CNN_BiLSTM_encoder(nn.Module):
    def __init__(self, inputDim, nhead=8, dropout=0.5, dropout_input=0.01, sequence_length=32,
                 specific='general', SEED=-1, out_target=None, noise=0.001, extra='', lr=0.01, lmd=0, CNN_out=32,
                 bi_lstm_layers=2, kernel=5, outputDim=4, out_len=7, auto_regressive=None, path=""):
        super(Parallel_CNN_BiLSTM_encoder, self).__init__()

        ######################
        # Model's parameters #
        ######################

        self.lr = lr
        self.lmd = lmd
        self.drop_prop = dropout
        self.drop_propIN = dropout_input
        self.noise = noise

        self.outputDim = outputDim
        self.inputDim = inputDim
        self.out_len = out_len
        self.inputLen = sequence_length

        self.path = path
        self.specific = specific
        self.extra = extra

        # Defining target time-steps
        if out_target is None:
            self.out_steps = [0, 1, 6]
        else:
            self.out_steps = out_target

        if SEED == -1:
            SEED = torch.randint(1, 10000000000000, (1,)).item()
        torch.manual_seed(SEED)

        ################################
        # Convolutional Neural Network #
        ################################

        # CNN Layers to discover patterns in our sequence
        # First CNN Block
        padding = int((kernel - 1) / 2)  # 0 padding strategy
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

        self.layer_norm_cnn = nn.LayerNorm(normalized_shape=CNN_out)

        ################################
        #         Encoder stack        #
        ################################

        # Embedding layer (Optional)
        # self.fc_encoder = nn.Sequential(
        #     nn.Linear(CNN_out, d_model),
        #     nn.ReLU()
        #)

        d_model = CNN_out

        ffnn_dim = 2 * d_model

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 3)

        ################################
        #         BiLSTM cell          #
        ################################

        BiLSTM = True
        self.bi_lstm = nn.LSTM(
            bidirectional=BiLSTM,  # Bi-LSTM setting
            input_size=CNN_out,
            hidden_size=CNN_out,
            num_layers=bi_lstm_layers,
            batch_first=True
        )
        LSTM_shape = 2 * CNN_out if BiLSTM else CNN_out
        self.layer_norm_LSTM = nn.LayerNorm(normalized_shape=LSTM_shape)

        ################################
        #        Output layer          #
        ################################

        self.final_shape = LSTM_shape + d_model

        self.fc_out = nn.Sequential(
            nn.Linear(self.final_shape, 64),
            nn.ReLU(),
            nn.Linear(64, outputDim),
        )

        # Dropout to improve generalization
        self.dropout = nn.Dropout(self.drop_prop)
        self.dropout_input = nn.Dropout(self.drop_propIN)

        #########################
        # Xavier initialization #
        #########################

        # CNN initialization
        temp_length = sequence_length
        for CNN in [self.cnn1, self.cnn2, self.cnn3]:
            for module in CNN:
                if not (isinstance(module, nn.Conv1d)):
                    continue
                temp_length, module = CNN_weight_init(module, temp_length)

        # Feed Forward Initialization
        for FC in [self.fc_out]:
            for module in FC:
                if not (isinstance(module, nn.Linear)):
                    continue
                module = FC_weight_init(module, temp_length)

        # Encoder initialization
        self.encoder = enc_dec_weight_init(self.encoder)

        ############################################################

    def forward(self, x, Y=None, training=None):

        batch = x.size(0)
        dim = x.size(2)

        # Input
        x = self.dropout_input(x)

        # Convolutional stack
        x = x.permute(0, 2, 1)  # (batch_size, steps, channels) -> (batch_size, channels, steps)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.permute(0, 2, 1)  # (batch_size, channels, steps) -> (batch_size, steps, channels)

        # Saving it to decrease the number of operations during the autoregressive cycle
        cnn_x = x.to(x.device)

        x = self.dropout(x)

        x_lstm, _ = self.bi_lstm(x)
        x_lstm = self.layer_norm_LSTM(x_lstm)

        x_encoder = self.encoder(x)

        x = torch.concatenate((x_lstm, x_encoder), dim=2)
        del x_encoder, x_lstm

        # Output batch with dimensionality (batch, out_len, outputDim)
        Y = torch.zeros((batch, self.out_len, self.outputDim)).to(x.device)
        Y[:, 0, :] = self.fc_out(x[:, [-1], :]).squeeze()

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
            x = self.layer_norm_cnn(cnn_x)

            # To effectively make use of the bidirectional recurrent layer,
            # past CNN outputs will be passed with the new sample "last" to generate the new output
            x_lstm, _ = self.bi_lstm(x)
            x_lstm = self.layer_norm_LSTM(x_lstm)

            x_encoder = self.encoder(x)

            x = torch.concatenate((x_lstm, x_encoder), dim=2)
            del x_encoder, x_lstm

            Y[:, t, :] = self.fc_out(x[:, [-1], :]).squeeze()
            last = torch.zeros((batch, 1, self.inputDim)).to(x.device)
            last[:, 0, :4] = Y[:, t, :]

        return Y

    def validation_loss(self, x, Y):
        with torch.no_grad():
            output = self(x)
            criterion = nn.MSELoss()
            loss = criterion(output, Y)
            return loss.item()
