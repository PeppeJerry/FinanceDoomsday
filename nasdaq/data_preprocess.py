import pandas as pd
import json
from models.utils import new_features, date2Timestamp, minmax_norm

###########################
# This is an example code #
###########################

# Selecting the block of datasets that will be normalized
block_target = "test"

# Reading JSON file for company information
with open('Companies.json', 'r') as file:
    c = json.load(file)
c_num = len(c[block_target])

# Loading datasets from "raw_data" folder
datasets = []
names = []
for i in range(c_num):
    names.append(c[block_target][str(i)])
    temp = "raw_data/" + names[i] + ".csv"
    datasets.append(pd.read_csv(temp).drop(columns=['Adj Close', 'Volume']).sort_values(by=['Date']))
c[block_target + "_normalized"] = {}

# preprocess data
for i in range(len(datasets)):
    datasets[i] = date2Timestamp(datasets[i])
    datasets[i] = new_features(datasets[i])
    datasets[i] = minmax_norm(datasets[i], names[i])

    # Saving normalized data into "processed_data" folder
    datasets[i].to_csv("processed_data/" + names[i] + ".csv", index=False)
    c[block_target + "_normalized"][str(i)] = names[i]

# Update "FinanceDoomsday/nasdaq/Companies.json" with new blocks of data
with open("Companies.json", "w") as outfile:
    json.dump(c, outfile, indent=4)
