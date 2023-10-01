import torch
from models.EncoderDecoder_Transformer import EncoderDecoder_Transformer
from models.CNN_BiLSTM import CNNBiLSTM
from models.Parallel_CNN_BiLSTM_encoder import Parallel_CNN_BiLSTM_encoder

import torch.nn as nn
from models.utils import *
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

# General parameters
N = 32
SEED = 242
pathTransformer = "models/EncoderDecoder_Transformer_Weights/"
pathLSTM = "models/CNN_BiLSTM_Weights/"
pathParallel = "models/Parallel_CNN_encoder_BiLSTM_Weights/"


def decreasingLR(lr, code=0):
    if code == 0:
        return lr / 2
    if code == 1:
        return lr / 5
    if code == 2:
        return lr / 10


def generate_permutations(model):
    # Defining hyperparameters to tune
    lr = [0.01, 0.005]
    lmd = [0.01, 0.001]
    dropout = [0.2, 0.5]
    dropout_input = [0, 0.01]
    noise = [0, 0.001]

    # Creating every possible combination
    perm = []
    for lr0 in lr:
        for lmd0 in lmd:
            for dropout0 in dropout:
                for dropout_input0 in dropout_input:
                    if model == "BiLSTM":
                        perm.append([lr0, lmd0, dropout0, dropout_input0, 0])
                    else:
                        for noise0 in noise:
                            perm.append([lr0, lmd0, dropout0, dropout_input0, noise0])
    return perm


def train_my_model(model, x_train, Y_train, x_validation, Y_validation, epochs=1000, batch_size=64, threshold=100):
    # Use CUDA if available as a process unit, CPU otherwise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # moving resources to the same device
    model = model.to(device)
    x_train = x_train.to(device)
    Y_train = Y_train.to(device)
    x_validation = x_validation.to(device)
    Y_validation = Y_validation.to(device)

    ####################################
    # Data loader for batch generation #
    ####################################

    allData = TensorDataset(x_train, Y_train)
    train_loader = DataLoader(allData, batch_size=batch_size, shuffle=True)
    batch_iter = iter(train_loader)

    # copy of the starting learning rate
    # NOTE: During the training process the learning rate can decrease if the model doesn't improve
    lr = model.lr

    # AdamW optimizer
    # Better L2 Regularization compared to Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model.lmd, betas=(0.9, 0.99))

    # MSE loss
    criterion = nn.MSELoss()

    # Start training
    model.train()

    # Best losses encountered during training
    loss_counter = float("inf")
    loss_val_counter = float("inf")

    # Loss history
    losses = []
    losses_val = []

    LowLR = 0
    decreased = 0
    for epoch in range(epochs):
        LowLR = LowLR + 1

        # Checking if iterator has a batch another batch to give
        try:
            x_batch, Y_batch = next(batch_iter)
        except StopIteration:
            allData = TensorDataset(x_train, Y_train)
            train_loader = DataLoader(allData, batch_size=batch_size, shuffle=True)
            batch_iter = iter(train_loader)
            x_batch, Y_batch = next(batch_iter)

        output = model(x_batch, Y_batch, training=True)

        # Standard MSE loss by just considering the target features are considered (Open, High, Low, Close)
        loss = criterion(output[:, :, :4], Y_batch[:, :, :4])

        # The training process will not consider these lines below for the back propagation and gradient computation
        with torch.no_grad():
            losses.append(loss.item())

            output = model(x_validation)
            loss_val = criterion(output[:, :, :4], Y_validation[:, :, :4])
            losses_val.append(loss_val.item())

            # Check if the new loss is less than the best loss ever recorded
            if loss.item() < loss_counter:
                LowLR = 0
                decreased = 0
                loss_counter = loss.item()
                print("Training   [{},{:.13f}]".format(epoch + 1, loss_counter))
                # Saving the model just in case
                torch.save(model.state_dict(), model.path + model.specific + model.extra + "_TRAIN" + '.pth')

            # Check if the new validation loss is less than the best loss ever recorded
            if loss_val.item() < loss_val_counter:
                LowLR = 0
                decreased = 0
                loss_val_counter = loss_val.item()
                print("Validation [{},{:.13f}]".format(epoch + 1, loss_val_counter))
                # Saving the model just in case
                torch.save(model.state_dict(), model.path + model.specific + model.extra + "_VAL" + '.pth')

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Decreasing Learning Rate if the model is not improving after "threshold" iterations
        # NOTE: "threshold" must be carefully chosen
        if LowLR >= threshold:
            decreased = decreased + 1
            LowLR = 0
            newLR = decreasingLR(lr, code=2)
            print("Lowering lr [{} -> {}]".format(lr, newLR))
            lr = newLR
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model.lmd, betas=(0.9, 0.99))

        # Even if after decreasing the learning rate the model still doesn't improve then stop the process
        if decreased >= 4:
            print("Too many decreased LR at epoch: [{}]".format(epoch))
            break

    model.eval()
    # Saving the last parameters
    torch.save(model.state_dict(), model.path + model.specific + model.extra + '.pth')

    best_val = min(losses_val)
    best_train = min(losses)

    last_val = losses_val[-1]
    last_train = losses[-1]

    return losses, losses_val, (best_train, best_val), (last_train, last_val)


def holdout():
    ###############################################################################################################
    # The result of this function will be three CSV files containing information about the best set of parameters #
    ###############################################################################################################

    # Data retrival
    training_set = "train_normalized"
    validation_set = "validation_normalized"
    x_train, Y_train = block_dataset(training_set, N=N)
    x_validation, Y_validation = block_dataset(validation_set, N=N)

    # Saving best parameter found
    best = []
    best_loss = []
    best_loss_val = []
    best_val = float("inf")
    best_train = float("inf")
    best_extra = ""

    for p in generate_permutations("BiLSTM"):
        extra = [str(p[i]) for i in range(len(p))]
        extra = "_".join(extra)
        extra = "_" + extra + "_"
        model = CNNBiLSTM(inputDim=x_train.size(2), out_target=[0, 1, 6], extra=extra,
                          specific="CNNBiLSTM", bi_lstm_layers=3,
                          lr=p[0], lmd=p[1], dropout=p[2], dropout_input=p[3], noise=p[4], path=pathLSTM
                          )
        losses, losses_val, (B_T, B_V), (L_T, L_V) = train_my_model(model, x_train, Y_train, x_validation, Y_validation,
                                                                    epochs=500, batch_size=128, threshold=100
                                                                    )
        # If a new set of the best hyperparameters has been found, they will be saved
        if L_V < best_val:
            best_extra = extra
            best_val = L_V
            best_train = L_T
            best_loss = losses
            best_loss_val = losses_val
            best = p
            print("New best set of parameters found! Loss validation: [{:.7f}]".format(best_val))
            print("[lr/lmd/drop/drop_in/noise] = [{}/{}/{}/{}/{}]".format(best[0], best[1], best[2], best[3], best[4]))

    columns = ["lr", "lmd", "dropout", "dropout_in", "noise", "best valid"]
    tot = [str(i) for i in range(len(best_loss))]
    tot = columns + tot
    data_val = [best[0], best[1], best[2], best[3], best[4], best_val] + best_loss_val
    data_train = [best[0], best[1], best[2], best[3], best[4], best_train] + best_loss
    data = np.array([data_val, data_train])

    # Saving information about the best set of hyperparameters
    df = pd.DataFrame(data, columns=tot)
    df.to_csv("BiLSTM_" + best_extra + ".csv", index=False)
    del data_train, data_val, data, df

    best_val = float("inf")
    for p in generate_permutations("Transformer"):
        extra = [str(p[i]) for i in range(len(p))]
        extra = "_".join(extra)
        extra = "_" + extra + "_"
        model = Parallel_CNN_BiLSTM_encoder(inputDim=x_train.size(2), out_target=[0, 1, 6], extra=extra,
                                            specific="Transformer", bi_lstm_layers=3, path=pathParallel,
                                            lr=p[0], lmd=p[1], dropout=p[2], dropout_input=p[3], noise=p[4],
                                            )
        losses, losses_val, (B_T, B_V), (L_T, L_V) = train_my_model(model, x_train, Y_train, x_validation, Y_validation,
                                                                    epochs=500, batch_size=128, threshold=100
                                                                    )
        # If a new set of the best hyperparameters has been found, they will be saved
        if L_V < best_val:
            best_extra = extra
            best_val = L_V
            best_train = L_T
            best_loss = losses
            best_loss_val = losses_val
            best = p
            print("New best set of parameters found! Loss validation: [{:.7f}]".format(best_val))
            print("[lr/lmd/drop/drop_in/noise] = [{}/{}/{}/{}/{}]".format(best[0], best[1], best[2], best[3], best[4]))

    columns = ["lr", "lmd", "dropout", "dropout_in", "noise", "best valid"]
    tot = [str(i) for i in range(len(best_loss))]
    tot = columns + tot
    data_val = [best[0], best[1], best[2], best[3], best[4], best_val] + best_loss_val
    data_train = [best[0], best[1], best[2], best[3], best[4], best_train] + best_loss
    data = np.array([data_val, data_train])

    # Saving information about the best set of hyperparameters
    df = pd.DataFrame(data, columns=tot)
    df.to_csv("BiLSTM_" + best_extra + ".csv", index=False)
    del data_train, data_val, data, df

    best_val = float("inf")
    for p in generate_permutations("Transformer"):
        extra = [str(p[i]) for i in range(len(p))]
        extra = "_".join(extra)
        extra = "_" + extra + "_"
        model = EncoderDecoder_Transformer(inputDim=x_train.size(2), out_target=[0, 1, 6], extra=extra,
                                           specific="Transformer", outputDim=4, bi_lstm_layers=3, path=pathTransformer,
                                           lr=p[0], lmd=p[1], dropout=p[2], dropout_input=p[3], noise=p[4]
                                           )
        losses, losses_val, (B_T, B_V), (L_T, L_V) = train_my_model(model, x_train, Y_train, x_validation, Y_validation,
                                                                    epochs=500, batch_size=128, threshold=100
                                                                    )
        # If a new set of the best hyperparameters has been found, they will be saved
        if L_V < best_val:
            best_extra = extra
            best_val = L_V
            best_train = L_T
            best_loss = losses
            best_loss_val = losses_val
            best = p
            print("New best set of parameters found! Loss validation: [{:.7f}]".format(best_val))
            print("[lr/lmd/drop/drop_in/noise] = [{}/{}/{}/{}/{}]".format(best[0], best[1], best[2], best[3], best[4]))

    columns = ["lr", "lmd", "dropout", "dropout_in", "noise", "best valid"]
    tot = [str(i) for i in range(len(best_loss))]
    tot = columns + tot
    data_val = [best[0], best[1], best[2], best[3], best[4], best_val] + best_loss_val
    data_train = [best[0], best[1], best[2], best[3], best[4], best_train] + best_loss
    data = np.array([data_val, data_train])

    # Saving information about the best set of hyperparameters
    df = pd.DataFrame(data, columns=tot)
    df.to_csv("BiLSTM_" + best_extra + ".csv", index=False)


def evaluate_my_model(model, test_set):
    # Data retrival
    x_test, Y_test = block_dataset(block_to_use=test_set, N=N)

    # Moving resources to the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    Y_test = Y_test.to(device)
    x_test = x_test.to(device)

    # Taking target features
    Y_test = Y_test[:, :, :4]

    # Prediction (torch.no_grad() is needed in order to avoid gradient computation on the set)
    with torch.no_grad():
        Y_pred = model(x_test)

    # Metrics evaluations
    mae = torch.mean(torch.abs(Y_test - Y_pred), dim=(0, 1))
    mse = torch.mean((Y_test - Y_pred) ** 2, dim=(0, 1))
    rmse = torch.sqrt(mse)
    smape = 2.0 * torch.mean(torch.abs(Y_test - Y_pred) / (torch.abs(Y_test) + torch.abs(Y_pred) + 1e-6),
                             dim=(0, 1)) * 100.0
    m_mae = torch.mean(mae)
    m_smape = torch.mean(smape)
    m_mse = torch.mean(mse)
    m_rmse = torch.mean(rmse)

    # Printing metrics
    channels = ["Open", "High", "Low", "Close"]
    for channel, i in zip(channels, range(len(channels))):
        c_mae = mae[i]
        c_smape = smape[i]
        c_mse = mse[i]
        c_rmse = rmse[i]
        print("Var: [{}] | MAE: [{:.6f}] | SMAPE: [{:.6f}] | MSE: [{:.6f}] | RMSE: [{:.6f}]".format(channel,
                                                                                                    c_mae, c_smape,
                                                                                                    c_mse, c_rmse))
    print("----------------------------------------------------------------------")
    print("Overall model performance")
    print("MAE: [{:.6f}] | SMAPE: [{:.6f}] | MSE: [{:.6f}] | RMSE: [{:.6f}]".format(m_mae, m_smape, m_mse, m_rmse))

    print("----------------------------------------------------------------------")


def predict_my_company(model, company):
    # Data retrival
    x_test, _ = specific_dataset(dataset_name=company + "_test", N=N)

    # Generate the real sequence without normalizing it
    x_real = pd.read_csv("nasdaq/raw_data/" + company + "_test.csv").drop(columns=['Adj Close', 'Volume']).sort_values(
        by=['Date'])
    x_real = date2Timestamp(x_real)
    x_real, Y_real = generate_sequence(x_real.to_numpy())

    # Moving resources to the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_test = x_test.to(device)

    # Taking target features
    # Prediction (torch.no_grad() is needed in order to avoid gradient computation on the set)
    with torch.no_grad():
        Y_norm = model(x_test)

    # Retrieving information about the normalization
    info = json.load(open("nasdaq/normalization/" + company + "_test.json"))
    min_c = np.array(info['min'])[:4]
    max_c = np.array(info['max'])[:4]

    # De-normalization of the prediction
    Y_norm = Y_norm.to("cpu").numpy()
    Y_pred = denormalize(Y_norm, min_vect=min_c, max_vect=max_c)
    Y_real = Y_real.numpy()

    OPEN = []
    HIGH = []
    LOW = []
    CLOSE = []

    # Generating pair (y, y') per feature for each target day
    for day in [0, 1, 6]:
        i = 0
        OPEN.append(np.hstack((Y_real[:, day, [i + 1]], Y_pred[:, day, [i]])))
        i = i + 1
        HIGH.append(np.hstack((Y_real[:, day, [i + 1]], Y_pred[:, day, [i]])))
        i = i + 1
        LOW.append(np.hstack((Y_real[:, day, [i + 1]], Y_pred[:, day, [i]])))
        i = i + 1
        CLOSE.append(np.hstack((Y_real[:, day, [i + 1]], Y_pred[:, day, [i]])))

    return OPEN, HIGH, LOW, CLOSE


def plot_variable(var, day=1, name="", company_name=""):
    title = "Generic plot"
    if name != "":
        title = "Plotting [" + name + "] feature"
    if company_name != "":
        title = title + " (" + company_name + ")"

    data = []
    if day == 1:  # 1-day prediction
        title = title + " 1-day predictions"
        data = var[0].transpose()
    if day == 2:  # 2-days prediction
        title = title + " 2-days predictions"
        data = var[1].transpose()
    if day == 7:  # 1-week prediction
        title = title + " 1-week prediction"
        data = var[2].transpose()
    if data is []:
        return
    plt.plot(data[0], label='Real values', color="black")
    plt.plot(data[1], label='Predictions', color="red", linestyle='-')

    plt.title(title)
    plt.xlabel("Consecutive days")
    plt.ylabel("Stock value")
    plt.legend()
    plt.show()


def examples():
    #####################################################################################
    # This code is not meant to be executed as it is.                                   #
    # It contains multiple examples in order to allow the user to create of Ad hoc code #
    #####################################################################################

    ###########
    # Holdout #
    ###########

    holdout()

    ########################
    # train a single model #
    ########################

    # Data retrival
    training_set = "train_normalized"
    validation_set = "validation_normalized"
    x_train, Y_train = block_dataset(training_set, N=N)
    x_validation, Y_validation = block_dataset(validation_set, N=N)

    specific = "CNNBiLSTM"
    model = CNNBiLSTM(inputDim=6, out_target=[0, 1, 6], outputDim=4,
                      specific=specific, auto_regressive=True, path=pathLSTM,
                      lr=0.005, lmd=0.001, dropout=0.2, dropout_input=0, noise=0, bi_lstm_layers=3
                      )
    train, val, _, _ = train_my_model(model, x_train, Y_train, x_validation, Y_validation, epochs=1000, batch_size=128,
                                      threshold=100)

    # Optional
    # Plot losses obtained during training
    plotLosses(train, val)

    ##################################
    # Loading already trained models #
    ##################################
    specific = "CNNBiLSTM"
    model = CNNBiLSTM(inputDim=6, out_target=[0, 1, 6], outputDim=4,
                      specific=specific, auto_regressive=True, path=pathLSTM,
                      lr=0.005, lmd=0.001, dropout=0.2, dropout_input=0, noise=0, bi_lstm_layers=3
                      )
    model.load_state_dict(torch.load(model.path + "/" + specific + ".pth"))

    ####################
    # Model evaluation #
    ####################

    evaluate_my_model(model, "test_normalized")

    #####################
    # Predict a company #
    #####################

    OPEN, HIGH, LOW, CLOSE = predict_my_company(model, "MSFT")

    # Optional
    # Plot predictions
    OPEN, HIGH, LOW, CLOSE = predict_my_company(model, "MSFT")
    plot_variable(OPEN, day=1, name="Open")  # Showing 1-day predictions
    plot_variable(OPEN, day=2, name="Open")  # Showing 2-days predictions
    plot_variable(OPEN, day=7, name="Open")  # Showing 1-week predictions


def main():
    specific = "CNNBiLSTM"
    model = CNNBiLSTM(inputDim=6, out_target=[0, 1, 6], outputDim=4,
                      specific=specific, auto_regressive=True, path=pathLSTM,
                      lr=0.005, lmd=0.001, dropout=0.2, dropout_input=0, noise=0, bi_lstm_layers=3
                      )
    model.load_state_dict(torch.load(model.path + "/" + specific + ".pth"))

    OPEN, HIGH, LOW, CLOSE = predict_my_company(model, "MSFT")

    plot_variable(OPEN, day=7, name="Open", company_name="Microsoft")


if __name__ == '__main__':
    main()
