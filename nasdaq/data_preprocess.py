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
def minmax_norm(data, name):
    columns = list(data.columns.values)
    n_array = data.drop(columns=["Date"]).values
    dates = data['Date'].values

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

    with open("normalization/" + name + ".json", "w") as outfile:
        json.dump({'norm': 'minmax', 'min': list(n_min), 'max': list(n_max)}, outfile, indent=4)
    return data


# New features that might improve the overall accuracy
def new_features(data):
    data['Volatility'] = (data['High'] - data['Low']) / data['High']
    data['Price_Change'] = (data['Close'] - data['Open']) / data['Close']
    return data


set_to_normalize = "general"
# Reading JSON file for company information
with open('Companies.json', 'r') as file:
    c = json.load(file)
c_num = len(c[set_to_normalize])

# Loading our datasets
datasets = []
names = []
for i in range(c_num):
    names.append(c[set_to_normalize][str(i)])
    temp = "raw_data/" + names[i] + ".csv"
    datasets.append(pd.read_csv(temp).drop(columns=['Adj Close']))  # Not relevant information
    # datasets.append(pd.read_csv(temp).drop(columns=["Volume"]))  # Volume does not need to be predicted

for i in range(len(datasets)):
    datasets[i] = date2Timestamp(datasets[i])
    datasets[i] = new_features(datasets[i])
    datasets[i] = minmax_norm(datasets[i], names[i])

# Generating new CSV files (These are the one that will be used)
for i in c[set_to_normalize]:
    datasets[int(i)].to_csv(c[set_to_normalize][i] + ".csv", index=False)
