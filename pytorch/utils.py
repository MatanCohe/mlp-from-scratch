import torch
import pandas as pd
import numpy as np


DEV = pd.read_csv('../data/validate.csv', header=None)
DEV = torch.from_numpy(DEV.values).to(torch.float)

TRAIN = pd.read_csv('../data/train.csv', header=None)
# normalize train data
mean = TRAIN.values[:, 1:].mean()
std = TRAIN.values[:, 1:].std()
# choose the columns indexes we wish to change
idx = np.arange(1, TRAIN.shape[1])
TRAIN[idx] = np.divide(TRAIN[idx] - mean, std)

TRAIN = torch.from_numpy(TRAIN.values).to(torch.float)

