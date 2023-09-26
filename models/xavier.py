import torch
import torch.nn as nn


def FC_weigth_init(module, sequence_length):
    # Evaluating fan-in (n_in) and fan-out (n_out)
    n_in = sequence_length * module.in_features
    n_out = sequence_length * module.out_features

    # Applying Xavier (Glorot) initialization
    weight_coefficient = torch.tensor(2 / (n_in + n_out))
    weight_coefficient = torch.sqrt(weight_coefficient).item()
    module.weight.data = torch.randn(module.weight.data.size()) * weight_coefficient
    module.bias.data = torch.randn(module.bias.data.size()) * weight_coefficient
    return module


def LSTM_weigth_init(module):
    for name, param in module.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
    return module


def CNN_weight_init(module, sequence_length):
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
    return temp_length, module
