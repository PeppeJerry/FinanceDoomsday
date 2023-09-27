import math

import torch
import torch.nn as nn
import torch.nn.init as init
from models.xavier import FC_weight_init, enc_dec_weight_init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(1, max_len, d_model)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)


class EncoderDecoder_Transformer(nn.Module):
    def __init__(self, inputDim, d_model=32, nhead=8, dropout=0.5, dropout_input=0.01, sequence_length=128, out_len=14,
                 specific='general', SEED=-1, out_target=None, outputDim=4, noise=0.001, extra='', lr=0.01, lmd=0,
                 auto_regressive=False):
        super(EncoderDecoder_Transformer, self).__init__()

        self.auto_regressive = auto_regressive
        self.device = torch.device("cpu")
        self.noise = noise
        self.outputLen = out_len
        self.inputLen = sequence_length
        self.extra = extra
        self.lr = lr
        self.lmd = lmd
        self.drop_prop = dropout
        self.drop_propIN = dropout_input
        self.specific = specific

        # Defining target time-steps
        if out_target is None:
            self.out_steps = [0, 6, 13]
        else:
            self.out_steps = out_target

        if SEED == -1:
            SEED = torch.randint(1, 10000000000000, (1,)).item()
        torch.manual_seed(SEED)

        ffnn_dim = d_model

        # Embedding layer
        self.fc_in = nn.Sequential(
            nn.Linear(inputDim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, d_model),
            nn.ReLU(),
        )

        # Layer norm for numerical stability
        self.layer_norm_inx = nn.LayerNorm(d_model)
        self.layer_norm_inY = nn.LayerNorm(d_model)

        # Output of encoder and decoder will be normalized
        self.positional_encoder = PositionalEncoding(d_model=d_model)
        self.positional_decoder = PositionalEncoding(d_model=d_model, max_len=self.outputLen)

        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)

        # First encoder-decoder pair
        self.encoder0 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 1)
        self.decoder0 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 1)

        # Second encoder-decoder pair
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 1)
        self.decoder1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 1)

        # Third encoder-decoder pair
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 1)
        self.decoder2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 1)

        # Dropout to improve generalization
        self.dropout = nn.Dropout(self.drop_prop)
        self.dropout_input = nn.Dropout(self.drop_propIN)

        # Output
        self.fc_out = nn.Linear(d_model, outputDim)

        #########################
        # Xavier initialization #
        #########################

        self.encoder0 = enc_dec_weight_init(self.encoder0)
        self.encoder1 = enc_dec_weight_init(self.encoder1)
        self.encoder2 = enc_dec_weight_init(self.encoder2)

        self.decoder0 = enc_dec_weight_init(self.decoder0)
        self.decoder1 = enc_dec_weight_init(self.decoder1)
        self.decoder2 = enc_dec_weight_init(self.decoder2)

        for module in self.fc_in:
            if not (isinstance(module, nn.Linear)):
                continue
            module = FC_weight_init(module, sequence_length)

        self.fc_out = FC_weight_init(self.fc_out, self.outputLen)

        ############################################################

    def forward(self, x, Y=None, encoder_decoder_memoryPair=False, auto_regressive=False, training=True):

        # This means that we are either making the algorithm autoregressive or we are making a prediction
        if Y is None or auto_regressive:
            auto_regressive = True
            Y = torch.zeros(x.size()[0], self.outputLen, x.size()[2]).to(self.device)

        # During training, a noise will be added to further improve generalization capabilities of the model
        if training:
            noise = torch.randn(Y.size()).to(self.device) * self.noise
            Y = Y + noise
            noise = torch.randn(x.size()).to(self.device) * self.noise
            x = x + noise

        # Embedding for Y
        Y = self.dropout_input(Y)
        Y = self.fc_in(Y)
        Y = self.positional_decoder(Y)
        Y = self.layer_norm_inY(Y)

        # Embedding for x
        x = self.dropout_input(x)
        x = self.fc_in(x)
        x = self.positional_encoder(x)
        x = self.layer_norm_inx(x)
        x = self.dropout(x)

        # Generate encoder memories that will be passed down to their corresponding decoder
        memory0 = self.encoder0(x)
        memory1 = self.encoder1(memory0)
        memory2 = self.encoder2(memory1)

        for t in range(self.outputLen):
            # Due to dimensionality issues, temp_Y will be given up to (t+1) sequences while masking (t+1) only
            # If masking is not performed the transformer performance will drastically decrease (Losses & Overfitting)
            temp_Y = self.dropout(Y[:, :t + 1, :])
            temp_Y[:, t, :] = torch.zeros(Y.size(0), Y.size(2)).to(temp_Y)  # Masking future position

            decoder2 = self.decoder2(temp_Y, memory2)
            decoder0 = torch.zeros((0, 0, 0))
            if not encoder_decoder_memoryPair:
                # Decoders use their corresponding encoder memories to improve accuracy
                decoder1 = self.decoder1(decoder2, memory1)
                decoder0 = self.decoder0(decoder1, memory0)
            else:
                # It is possible to pass the last memory generated by the last encoder to all the stack of decoders
                # This might improve overfitting situations
                decoder1 = self.decoder1(decoder2, memory2)
                decoder0 = self.decoder0(decoder1, memory2)

            if auto_regressive:  # During prediction the output at t of the decoder will become its new input
                Y[:, t, :] = decoder0[:, -1, :]

        # Generating output
        Y = self.layer_norm_inY(Y)
        Y = self.dropout(Y)
        Y = self.fc_out(Y)

        return Y

    def train_model(self, train_loader, x_valid, Y_valid, epochs=300,
                    path="models/EncoderDecoder_Transformer_Weights/", threshold=50):
        # Defining if CUDA is available for processing, use CPU otherwise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = self.lr
        self.to(device)
        self.switch_CUDA()

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

            output = self(x_batch, Y_batch, training=True, auto_regressive=self.auto_regressive)

            # Standard loss but just the first 4 features are considered (Open, Close, High, Low) for better accuracy
            loss = criterion(output[:, :, :4], Y_batch[:, :, :4])

            # The training process will not consider these lines below for the back propagation
            with torch.no_grad():
                # We are taking the loss of the clean input x (without noise)
                output = self(x_batch, Y_batch, training=False)
                loss_temp = criterion(output[:, :, :4], Y_batch[:, :, :4])
                losses.append(loss_temp.item())

                output = self(x_valid, training=False)
                loss_val = criterion(output[:, :, :4], Y_valid[:, :, :4])
                losses_val.append(loss_val.item())

                # Check if the new loss is less than the best loss so far
                if loss_temp.item() < loss_counter:
                    LowLR = 0
                    loss_counter = loss_temp.item()
                    print("Training   [{},{:.13f}]".format(epoch + 1, loss_counter))

                # Check if the new validation loss is less than the best loss so far
                # Saving best parameters based on validation set
                if loss_val.item() < loss_val_counter:
                    LowLR = 0
                    loss_val_counter = loss_val.item()
                    print("Validation [{},{:.13f}]".format(epoch + 1, loss_val_counter))
                    torch.save(self.state_dict(), path + self.specific + self.extra + '.pth')

            # Backprop
            loss.backward()
            optimizer.step()

            # Decreasing Learning Rate if the model is not improving after "threshold" iterations
            # NOTE: "threshold" must be carefully chosen
            if LowLR >= threshold:
                LowLR = 0
                print("Lowering lr [{} -> {}]".format(lr, lr / 10))
                lr = lr / 10
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.lmd)

        # Taking the best parameters obtained for the validation set
        self.load_state_dict(torch.load(path + self.specific + self.extra + '.pth'))
        self.eval()
        self.switch_CPU()
        return losses, losses_val

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
