import pandas as pd
import json
import time
import datetime
import numpy as np


# This function converts a date from format "YYYY-MM-DD" to its timestamp equivalent
def date2Timestamp(data):
    data['Date'] = data['Date'].apply(
        lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())
    )
    return data


# min-max normalization
def minmax_norm(data, data_name):
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

    with open("normalization/" + data_name + ".json", "w") as outfile:
        json.dump({'norm': 'minmax', 'min': list(n_min), 'max': list(n_max)}, outfile, indent=4)
    return data


# New features that might improve the overall accuracy
def new_features(data):
    data['Volatility'] = (data['High'] - data['Low']) / data['High']
    data['Price_Change'] = (data['Close'] - data['Open']) / data['Close']
    return data


###########################
# This is an example code #
###########################

# Selecting the block of datasets that will be normalized together
block_target = "1Y_target"

# Reading JSON file for company information
with open('Companies.json', 'r') as file:
    c = json.load(file)
c_num = len(c[block_target])
c_years = c[block_target + "_years"]
gap = 2

# Loading our datasets
datasets = []
names = []
for i in range(c_num):
    names.append(c[block_target][str(i)])
    temp = "raw_data/" + names[i] + ".csv"
    datasets.append(pd.read_csv(temp).drop(columns=['Adj Close', 'Volume']).sort_values(by=['Date']))
c[block_target + "_normalized"] = {}

# pre-process stage
for i in range(len(datasets)):
    datasets[i] = date2Timestamp(datasets[i])
    datasets[i] = new_features(datasets[i])

    # Setting 1Y window size
    set_len = len(datasets[i])
    window_len = int(set_len / c_years)

    # If c_years == 1 then with this check will be normalized entirely without dividing it
    if c_years == 1: c_years = 2

    for j in range(c_years - 1):
        a = j * window_len

        # Last iteration will consider all the remaining samples ( 2 * window_len + residual samples )
        if j + 1 == c_years - 1:
            b = set_len
        else:
            b = (gap + j) * window_len

        # Temporary dataframe to store 2 years worth of samples
        # Each iteration will slide 1 year at a time creating overlaps of 1 year among normalizations
        temp = datasets[i][a:b].copy()
        temp = minmax_norm(temp, names[i] + "_" + str(j))
        name = names[i] + "_" + str(j)
        temp.to_csv("processed_data/" + name + ".csv", index=False)
        c[block_target + "_normalized"][str(100 * i + j)] = name

# Update "FinanceDoomsday/nasdaq/Companies.json" with new blocks of data
with open("Companies.json", "w") as outfile:
    json.dump(c, outfile, indent=4)
