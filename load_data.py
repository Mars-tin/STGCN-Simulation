import torch
import numpy as np
import pandas as pd


def load_matrix(file_path):
    """
    load matrix data from a csv file.
    """
    return pd.read_csv(file_path, header=None).values.astype(float)


def load_data(file_path, len_train, len_val, len_test=None):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    if len_test is not None:
        test = df[len_train + len_val:len_train + len_val + len_test]
    else:
        test = df[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    y = np.zeros([n_day * n_slot, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
            y[t] = data[e + n_pred - 1]
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
