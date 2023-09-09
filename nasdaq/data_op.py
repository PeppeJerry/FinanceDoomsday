import pandas as pd
import json
import time
import datetime
import numpy as np


# This function converts a date from format "YYYY-MM-DD" to its timestamp equivalent
def date2Timestamp(data):
    for pos in range(len(data)):
        data[pos]['Date'] = data[pos]['Date'].apply(
            lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())
        )
    return data


# min-max normalization
def minmax_norm(data):
    global n_min, n_max
    n_array = np.empty([len(data), data[0].shape[0], data[0].shape[1] - 1])  # "-1" because we won't consider "Date"
    dates = np.empty([len(data), data[0].shape[0]])
    columns = list(data[0].columns.values)
    for pos in range(len(data)):
        n_array[pos, :, :] = data[pos].drop(columns=["Date"]).values
        dates[pos, :] = data[pos]['Date'].values

        # Generating min and max matrix, each row[i] represents min and max values for the dataset "i"
        n_min = np.min(n_array, axis=1)
        n_max = np.max(n_array, axis=1)

    # Data is not necessary from this point on
    del data

    # Generating min and max array among all datasets
    n_min = np.min(n_min, axis=0)
    n_max = np.max(n_max, axis=0)

    # Applying min-max
    for pos in range(n_array.shape[0]):
        n_array[pos, :, :] = (n_array[pos, :, :] - n_min) / (n_max - n_min)

    # Generating new DataFrames with normalized data
    n_array = np.dstack((dates, n_array))
    data = []
    for pos in range(n_array.shape[0]):
        data.append(pd.DataFrame(n_array[pos, :, :], columns=columns))
    del n_array

    # Keeping information about what kind of normalization and all parameters involved
    # Since prediction y# is normalized, thanks to these data it will be denormalized in y_ as it follows:
    # y_ = y# * (max - min) + min
    with open("normalization.json", "w") as outfile:
        json.dump({
            'norm': 'minmax',
            'min': list(n_min),
            'max': list(n_max)
        }, outfile, indent=4)
    return data

# One-hot encoding for each company
def one_hotEncoding(data, num):
    for i in range(num):
        for j in range(num):
            value = 0
            if i == j:
                value = 1
            data[i][c['encoding'][str(j)]] = value
    return data


# Reading JSON file for company information
c = json.load(open("Companies.json"))
c_num = len(c['encoding'])

# Loading our datasets
datasets = []
for i in range(c_num):
    temp = "raw_data/" + c['encoding'][str(i)] + ".csv"
    datasets.append(pd.read_csv(temp))

datasets = date2Timestamp(data=datasets)
datasets = minmax_norm(data=datasets)
datasets = one_hotEncoding(data=datasets, num=c_num)

# Generating new CSV files (These are the one that will be used)
for i in c['encoding']:
    datasets[int(i)].to_csv(c['encoding'][i] + ".csv", index=False)
