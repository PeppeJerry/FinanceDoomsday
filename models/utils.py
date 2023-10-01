import json
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# generating absolute paths to the project
abs_PATH = os.path.dirname(os.path.abspath(__file__)).split("\\")
project_PATH = abs_PATH[0]
for i in range(1, len(abs_PATH) - 1):
    project_PATH = project_PATH + "/" + abs_PATH[i]
del abs_PATH


# This function takes multiple datasets from multiple files and creates a unique list of sequences for all of them
# Note: FinanceDoomsday/nasdaq/Companies.json contains all the needed information (names)
# Datasets are placed inside "FinanceDoomsday/nasdaq/raw_data/*"
def block_dataset(block_to_use="general", out_steps=None, N=128):
    # Reading JSON file for company information
    if out_steps is None:
        out_steps = [0, 1, 6]

    c = json.load(open(project_PATH + "/nasdaq/Companies.json"))

    # Sequences vectors
    x = []
    Y = []
    names = []

    # max_target represents the longest step to predict
    max_target = max(out_steps) + 1
    for file in c[block_to_use]:
        names.append(c[block_to_use][file])
        path = project_PATH + "/nasdaq/processed_data/" + c[block_to_use][file] + ".csv"
        data = pd.read_csv(path).sort_values(by=['Date']).drop(columns="Date")
        # We want to generate sequences that have proper target prediction at max target steps
        # In this project we want sequences with one of the target being at 2 weeks (14 time-steps later)
        temp_x = []
        temp_Y = []
        for j in range(data.shape[0] - N - max_target):
            temp_x.append(data.iloc[j:j + N].values)
            temp_Y.append(data.iloc[j + N:j + N + max_target].values)
            # Data are converted into tensors (list -> numpy obj -> tensor)
        x.append(torch.tensor(np.array(temp_x), dtype=torch.float32))
        Y.append(torch.tensor(np.array(temp_Y), dtype=torch.float32))

    # For the sake of debug, also because there variables are no longer needed
    del data, j, max_target, path, names, block_to_use, file, c

    # "x" & "Y" are lists of 3D tensors per single list, so they might be interpreted as 4D models overall
    # "x_" & "Y_" will reduce the dimensionality from 4D to 3D by concatenating all different "x_set"s and "Y_set"s
    # Stacking along the first axis (set to 0)
    x_ = torch.zeros(0, x[0].size()[1], x[0].size()[2])
    Y_ = torch.zeros(0, Y[0].size()[1], Y[0].size()[2])

    for x_set, Y_set in zip(x, Y):
        x_ = torch.cat((x_, x_set))
        Y_ = torch.cat((Y_, Y_set))

    # The results are two 3D tensors having dimensionality (Sequences, samples_per_sequences, features)
    return x_, Y_


# This function takes a specific dataset given the name of the file and creates sequences
# Specific case of "block_dataset"
def specific_dataset(dataset_name, out_steps=None, N=128):
    if out_steps is None:
        out_steps = [0, 1, 6]
    # Reading JSON file for company information
    c = json.load(open(project_PATH + "/nasdaq/Companies.json"))

    # Generating sequences
    x = []
    Y = []

    # max_target represents the longest time step to predict
    max_target = max(out_steps) + 1
    path = project_PATH + "/nasdaq/processed_data/" + dataset_name + ".csv"
    data = pd.read_csv(path).sort_values(by=['Date']).drop(columns="Date")
    # We want to generate sequences that have proper target prediction at max target steps
    # In this project we want sequences with one of the target being at 2 weeks (14 time-steps later)
    for j in range(data.shape[0] - N - max_target):
        x.append(data.iloc[j:j + N].values)
        Y.append(data.iloc[j + N:j + N + max_target].values)
        # x & Y are converted into pytorch tensors (list -> numpy obj -> tensor)
    del data

    x = torch.tensor(np.array(x), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    return x, Y


def generate_sequence(x, out_len=7, N=32):
    temp_x = []
    temp_Y = []
    # We want to generate sequences that have proper target prediction at max target steps
    # In this project we want sequences with one of the target being at 2 weeks (14 time-steps later)
    for j in range(x.shape[0] - N - out_len):
        temp_x.append(x[j:j + N])
        temp_Y.append(x[j + N:j + N + out_len])

    x = torch.tensor(np.array(temp_x), dtype=torch.float32)
    Y = torch.tensor(np.array(temp_Y), dtype=torch.float32)

    return x, Y


# This functions just summarize blocks of instructions that are commonly used together
def training_model(model, x_tr, Y_tr, x_v, Y_v, max_iter=200):
    # I know, this function is pretty much useless
    losses, losses_val = model.train_model(x_train=x_tr, Y_train=Y_tr, x_valid=x_v, Y_valid=Y_v, epochs=max_iter)
    return losses, losses_val


def plotLosses(losses, losses_val):
    # using plots to visualize losses together while training
    plt.plot(range(len(losses)), losses, label='Training Loss', color='blue')
    plt.plot(range(len(losses_val)), losses_val, label='Validation Loss', color='red')
    plt.title("Error graph")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


# Using normalization info inside "FinanceDoomsday/nasdaq/normalization/*" data are denormalized
def denormalize(x_norm, norm="minmax", min_vect=6, max_vect=9, mean=4, std=2):
    if norm == "minmax":
        x_pred = np.zeros(x_norm.shape)
        for i in range(x_norm.shape[0]):
            x_pred[i, :, :] = x_norm[i, :, :] * (max_vect - min_vect) + min_vect
        return x_pred


def date2Timestamp(data):
    data['Date'] = data['Date'].apply(
        lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())
    )
    return data


# min-max normalization
def minmax_norm(data, data_name, save_info=True):
    columns = list(data.columns.values)
    n_array = data.drop(columns=["Date"]).values
    dates = data['Date'].values
    del data

    # Generating min and max arrays
    n_min = np.min(n_array, axis=0)
    n_max = np.max(n_array, axis=0)

    # Applying min-max
    n_array = (n_array - n_min) / (n_max - n_min)

    # Generating new DataFrames with normalized data
    n_array = np.c_[dates, n_array]
    data = pd.DataFrame(n_array, columns=columns)
    del n_array

    # Keeping information about what kind of normalization and all parameters involved
    # Since prediction y# is normalized, thanks to these data it will be denormalized in y_ as it follows:
    # y_ = y# * (max - min) + min
    if save_info:
        with open("normalization/" + data_name + ".json", "w") as outfile:
            json.dump({'norm': 'minmax', 'min': list(n_min), 'max': list(n_max)}, outfile, indent=4)
    return data


# Feature addition
def new_features(data):
    data['Volatility'] = (data['High'] - data['Low']) / data['High']
    data['Price_Change'] = (data['Close'] - data['Open']) / data['Close']
    return data
