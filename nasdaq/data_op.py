import pandas as pd
import json
import time
import datetime
import numpy as np


# This function converts a date from format "YYYY-MM-DD" to its timestamp equivalent
def date2Timestamp(data):
    for i in range(len(data)):
        data[i]['Date'] = data[i]['Date'].apply(
            lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())
        )
    return data


# min-max normalization
def minmax_norm(data):
    n_array = np.empty([len(data), data[0].shape[0], data[0].shape[1] - 1])  # "-1" because we won't consider "Date"
    n_min = np.empty([len(data), data[0].shape[1] - 1])
    n_max = np.empty([len(data), data[0].shape[1] - 1])
    dates = np.empty([len(data), data[0].shape[0]])
    columns = list(data[0].columns.values)
    for i in range(len(data)):
        n_array[i, :, :] = data[i].drop(columns=["Date"]).values
        dates[i, :] = data[i]['Date'].values

        # Generating min and max matrix, each row[i] represents min and max values for the dataset "i"
        n_min = np.min(n_array, axis=1)
        n_max = np.max(n_array, axis=1)

    # Data is not necessary from this point on
    del data

    # Applying min-max
    for i in range(n_array.shape[0]):
        n_array[i, :, :] = (n_array[i, :, :] - n_min[i, :]) / (n_max[i, :] - n_min[i, :])

    # Generating new DataFrames with normalized data
    n_array = np.dstack((dates, n_array))
    data = []
    for i in range(n_array.shape[0]):
        data.append(pd.DataFrame(n_array[i, :, :], columns=columns))
    del n_array

    # Keeping information about what kind of normalization and all parameters involved
    # Since prediction y# is normalized, thanks to these data it will be denormalized in y_ as it follows:
    # y_ = y# * (max - min) + min
    info_min = []
    info_max = []
    for a_min,a_max in zip(n_min, n_max):
        info_min.append(list(a_min))
        info_max.append(list(a_max))
    del n_min, n_max

    with open("normalization.json", "w") as outfile:
        json.dump({
            'norm': 'minmax',
            'min': info_min,
            'max': info_max
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
