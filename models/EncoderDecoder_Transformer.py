import math

import torch
import torch.nn as nn
import torch.nn.init as init
from models.xavier import FC_weight_init, enc_dec_weight_init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32, dropout=0):
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
    def __init__(self, inputDim, outputDim=4, dropout=0.5, lr=0.01, lmd=0, dropout_input=0.01, noise=0.001,
                 sequence_length=32, out_len=7, SEED=-1, bi_lstm_layers=None,
                 auto_regressive=True, specific='general', extra='', out_target=None, path=""):
        super(EncoderDecoder_Transformer, self).__init__()

        ######################
        # Model's parameters #
        ######################

        d_model = 16
        nhead = 8

        self.lr = lr
        self.lmd = lmd
        self.noise = noise
        self.drop_prop = dropout
        self.drop_propIN = dropout_input

        self.auto_regressive = auto_regressive

        self.outputDim = outputDim
        self.outputLen = out_len
        self.inputLen = sequence_length

        self.path = path
        self.extra = extra
        self.specific = specific

        # Defining target time-steps
        if out_target is None:
            self.out_steps = [0, 1, 6]
        else:
            self.out_steps = out_target

        if SEED == -1:
            SEED = torch.randint(1, 10000000000000, (1,)).item()
        torch.manual_seed(SEED)

        ##############################
        # Input and target embedding #
        ##############################

        # Embedding layer
        self.fc_in = nn.Sequential(
            nn.Linear(inputDim, 8),
            nn.ReLU(),
            nn.Linear(8, d_model),
            nn.ReLU(),
        )

        # Positional encoding for both input and target sequence
        self.positional_encoder = PositionalEncoding(d_model=d_model, max_len=self.inputLen)
        self.positional_decoder = PositionalEncoding(d_model=d_model, max_len=self.outputLen)

        # Layer norm for numerical stability
        self.layer_norm_inx = nn.LayerNorm(d_model)
        self.layer_norm_inY = nn.LayerNorm(d_model)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)

        ##############################
        # Encoder and decoder stacks #
        ##############################

        ffnn_dim = 2 * d_model

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 3)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, ffnn_dim, self.drop_prop, batch_first=True), 3)

        # Dropout
        self.dropout = nn.Dropout(self.drop_prop)
        self.dropout_input = nn.Dropout(self.drop_propIN)

        # Output
        self.fc_out = nn.Linear(d_model, outputDim)

        #########################
        # Xavier initialization #
        #########################

        self.encoder = enc_dec_weight_init(self.encoder)
        self.decoder = enc_dec_weight_init(self.decoder)

        for module in self.fc_in:
            if not (isinstance(module, nn.Linear)):
                continue
            module = FC_weight_init(module, sequence_length)
        self.fc_out = FC_weight_init(self.fc_out, self.outputLen)

    def forward(self, x, Y=None, training=True, auto_regressive=True):

        # During training, the decoder stack can behave either self-conditioned or autoregressive
        if training:
            auto_regressive = self.auto_regressive

        # During training, a noise will be added to further improve generalization capabilities of the model
        if training and self.noise > 0:
            noise = torch.randn(x.size()).to(x.device) * self.noise
            x = x + noise

        # Embedding for x
        x = self.dropout_input(x)
        x = self.fc_in(x)
        x = self.positional_encoder(x)

        # Generate encoder memories that will be passed down to their corresponding decoder
        memory = self.encoder(x)
        output = torch.zeros(x.size()[0], self.outputLen, x.size()[2]).to(x.device)

        if Y is None or auto_regressive:
            auto_regressive = True
            Y = torch.zeros(x.size()[0], self.outputLen, x.size()[2]).to(x.device)
        else:
            Y = self.fc_in(Y)

        for t in range(self.outputLen):
            # Due to dimensionality issues, temp_Y will be given up to (t) samples while masking (t) only
            # If masking is not performed the transformer performance will drastically decrease (Losses & Overfitting)
            temp_Y = self.dropout(Y[:, :t + 1, :])
            temp_Y[:, t, :] = torch.zeros(Y.size(0), Y.size(2)).to(temp_Y)  # Masking future position

            decoder = self.decoder(temp_Y, memory)
            output[:, t, :] = decoder[:, -1, :]
            if auto_regressive:  # During prediction the output at t of the decoder will become its new input
                Y[:, t, :4] = self.fc_out(decoder[:, -1, :]).squeeze()

        output = self.dropout(output)
        output = self.fc_out(output)
        return output
