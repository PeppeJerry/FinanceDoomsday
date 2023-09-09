import torch
import pandas as pd
import json
import numpy as np


def main():
    print('Where it all began')
    print(torch.__version__)

    # Reading JSON file for company information
    c = json.load(open("nasdaq/Companies.json"))
    c_num = len(c['encoding'])

    # Loading our datasets
    data = []
    for i in range(c_num):
        temp = "nasdaq/" + c['encoding'][str(i)] + ".csv"
        data.append(pd.read_csv(temp))
    data = data



if __name__ == '__main__':
    main()
