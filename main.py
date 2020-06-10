import random
import os
import zipfile
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from stgcn import *
from GaussianCopula import CopulaLoss

# Set random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

if (not os.path.isfile("data/W_228.csv")
        or not os.path.isfile("data/V_228.csv")):
    with zipfile.ZipFile("data/PeMS-M.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
matrix_path = "data/W_228.csv"
data_path = "data/V_228.csv"
save_path = "save/model.pt"

# Parameters
day_slot = 288
n_train, n_val, n_test = 34, 5, 5

n_his = 12
n_pred = 3
n_route = 228
Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0

batch_size = 50
epochs = 50
lr = 1e-3

loss_function = "mse"
tau_list = [0.01, 0.1, 1]
gamma_list = [0.1, 1]

# Graph
W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
Lk = torch.from_numpy(Lk).float().to(device)

X = load_matrix(data_path)
# A = get_adjecency_matrix(W, X)

# Standardization
train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

# Transformation
x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

# Data Loader
train_data = data.TensorDataset(x_train, y_train)
train_iter = data.DataLoader(train_data, batch_size, shuffle=True)
val_data = data.TensorDataset(x_val, y_val)
val_iter = data.DataLoader(val_data, batch_size)
test_data = data.TensorDataset(x_test, y_test)
test_iter = data.DataLoader(test_data, batch_size)

# Loss_fn, criterion, model, optimizer and LR scheduler
loss_fn = nn.MSELoss()
if loss_function == "copula":
    loss_fn = CopulaLoss()
criterion = nn.MSELoss()
model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

df = {"tau": [], "gamma": [], "MAE": [], "MAPE": [], "RMSE": []}
if loss_function == "mse":
    tau_list = [0.1]
    gamma_list = [0.1]

for tau in tau_list:
    for gamma in gamma_list:
        df["tau"].append(tau)
        df["gamma"].append(gamma)
        # sigma = get_covariance(A, tau, gamma)
        sigma = None

        # Training
        min_val_loss = np.inf
        for epoch in range(1, epochs + 1):
            loss_sum, n = 0.0, 0
            model.train()
            for x, y in train_iter:
                y_pred = model(x).view(len(x), -1)
                if hasattr(loss_fn, 'requires_cov'):
                    loss = loss_fn(y_pred, y, sigma)
                else:
                    loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = criterion(y_pred, y).item()
                loss_sum += loss_val * y.shape[0]
                n += y.shape[0]
            scheduler.step()
            val_loss = evaluate_model(model, loss_fn, val_iter, sigma)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
            print("epoch", epoch, ", train loss:", loss_sum / n, ", validation loss:", val_loss)

        best_model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
        best_model.load_state_dict(torch.load(save_path))

        loss = evaluate_model(best_model, loss_fn, test_iter, sigma)
        MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
        print("test loss:", loss, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
        df["MAE"].append(MAE)
        df["MAPE"].append(MAPE)
        df["RMSE"].append(RMSE)

pd_writer = pd.DataFrame(df)
pd_writer.to_csv('output.csv', index=False, header=False)
