import torch
import pandas as pd


DEV = pd.read_csv('../data/validate.csv', header=None)
DEV = torch.from_numpy(DEV.values).to(torch.float)
TRAIN = pd.read_csv('../data/train.csv', header=None)
TRAIN = torch.from_numpy(TRAIN.values).to(torch.float)

