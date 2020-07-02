import random
import os
import zipfile
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from stgcn import *
from gaussian_copula import CopulaLoss
from get_covariance import get_covariance

# CUDNN setup
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

# Set random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Parameters
day_slot = 288
n_train, n_val, n_test = 34, 5, 5

# M: the previous M traffic observations
n_his = 12
# H: the next H time steps(= 15*H min) to predict
n_pred = 3
# n: Number of monitor stations
n_route = 228
# K; Kernel size of spatial and temporal blocks
Ks, Kt = 3, 3
# bs: channel configs of ST-Conv blocks.
blocks = [[1, 32, 64], [64, 32, 128]]
# p: Drop rate of drop out layer
drop_prob = 0
rho = 0.02

# Get files ready
matrix_path = "data/W_228.csv"
data_path = "data/V_228.csv"
adj_path = "data/cov_228.csv"
save_path = "save/model.pt"
resolution = 100

if (not os.path.isfile(matrix_path)
        or not os.path.isfile(data_path)):
    with zipfile.ZipFile("data/PeMS-M.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
if not os.path.isfile(adj_path):
    get_covariance(rho=rho, resolution=resolution, n_his=n_his)


batch_size = 50
epochs = 50
lr = 1e-3

loss_function = "copula"
density = 30
tau_list = [0.01, 0.1, 1]
gamma_list = [0.1, 1]

# Graph
# W: weight adjacency matrix
W = load_matrix(matrix_path)
# L: Rescaled laplacian
L = scaled_laplacian(W)
# Theta: Kernel (ks, n_route, n_route)
Lk = cheb_poly(L, Ks)
Lk = torch.from_numpy(Lk).float().to(device)
# cov: covariance of training data
cov = load_matrix(adj_path).reshape((-1, n_route, n_route))
cov = torch.tensor(cov).to(device)

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
train_iter = data.DataLoader(train_data, batch_size, shuffle=False)
val_data = data.TensorDataset(x_val, y_val)
val_iter = data.DataLoader(val_data, batch_size)
test_data = data.TensorDataset(x_test, y_test)
test_iter = data.DataLoader(test_data, batch_size)

# Loss_fn, criterion, model, optimizer and LR scheduler
loss_fn = nn.MSELoss()
if loss_function == "copula":
    loss_fn = CopulaLoss()
criterion = nn.MSELoss()
model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device=device)
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

        # Training
        min_val_loss = np.inf
        for epoch in range(1, epochs + 1):
            loss_sum, n = 0.0, 0
            model.train()
            iter = 0
            for x, y in train_iter:
                y_pred = model(x).view(len(x), -1)
                if hasattr(loss_fn, 'requires_cov'):
                    sigma = cov[iter * batch_size // resolution]
                    iter += 1
                    loss = loss_fn(y_pred, y, sigma)
                    loss = torch.sum(loss)
                else:
                    loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = criterion(y_pred, y).item()
                loss_sum += loss_val * y.shape[0]
                n += y.shape[0]
            scheduler.step()
            val_loss = evaluate_model(model, loss_fn, val_iter, cov, resolution, batch_size)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
            print("epoch", epoch, ", train loss:", loss_sum / n, ", validation loss:", val_loss)

        best_model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
        best_model.load_state_dict(torch.load(save_path))

        loss = evaluate_model(best_model, loss_fn, test_iter, cov, resolution, batch_size)
        MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
        print("test loss:", loss, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
        df["MAE"].append(MAE)
        df["MAPE"].append(MAPE)
        df["RMSE"].append(RMSE)

pd_writer = pd.DataFrame(df)
pd_writer.to_csv('output.csv', index=False, header=False)
