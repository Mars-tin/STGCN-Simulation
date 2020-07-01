import os
import zipfile

from sklearn.preprocessing import StandardScaler
from load_data import *


# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Unzip data
if (not os.path.isfile("data/W_228.csv")
        or not os.path.isfile("data/V_228.csv")):
    with zipfile.ZipFile("data/PeMS-M.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
matrix_path = "data/W_228.csv"
data_path = "data/V_228.csv"
save_path = "data/A_228.csv"
x_temp_path = "data/cov/x.csv"
w_temp_path = "data/cov/w.csv"

# Parameters
day_slot = 288
n_train, n_val, n_test = 34, 5, 5
n_his = 12
n_pred = 3
n_route = 228
rho = 0.02

# Transform data
train, _, _ = load_data(data_path, n_train * day_slot, n_val * day_slot)
scaler = StandardScaler()
train = scaler.fit_transform(train)
x_train, _ = data_transform(train, n_his, n_pred, day_slot, device)
n_x = x_train.shape[0]

# Process data
f = open(save_path, 'ab')
for i, x in enumerate(x_train):
    x = x.squeeze()
    x = x.t().matmul(x).numpy()
    np.savetxt(x_temp_path, x, delimiter=",")
    cmd = f'Rscript get_cov.R {x_temp_path} {w_temp_path} {rho} {n_his}'
    print('Processing the {}th x in x_train...'.format(i))
    os.system(cmd)
    cov = load_matrix(w_temp_path)
    cov = np.asarray(cov).squeeze()
    np.savetxt(f, cov, delimiter=",")

os.remove(x_temp_path)
os.remove(w_temp_path)
