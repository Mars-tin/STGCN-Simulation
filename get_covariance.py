import os
import zipfile

from sklearn.preprocessing import StandardScaler
from load_data import *


def get_covariance(rho=0.02, resolution=20, n_his=12):
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Unzip data
    matrix_path = "data/W_228.csv"
    data_path = "data/V_228.csv"
    save_path = "data/cov_228.csv"
    x_temp_path = "data/cov/x.csv"
    w_temp_path = "data/cov/w.csv"

    if (not os.path.isfile(matrix_path)
            or not os.path.isfile(data_path)):
        with zipfile.ZipFile("data/PeMS-M.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    if (os.path.isfile(save_path)):
        os.remove(save_path)

    # Parameters
    day_slot = 288
    n_train, n_val, n_test = 34, 5, 5
    n_pred = 3

    # Transform data
    train, _, _ = load_data(data_path, n_train * day_slot, n_val * day_slot)
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    x_train, _ = data_transform(train, n_his, n_pred, day_slot, device)
    n_x = x_train.shape[0]

    # Process data
    f = open(save_path, 'ab')
    for i in range(n_x // resolution + 1):
        start = i*resolution
        end = min(n_x, (i+1)*resolution)
        idx = (start + end) // 2
        x = x_train[idx]

        x = x.squeeze().to(device="cpu")
        # x = x.t().matmul(x).numpy()
        np.savetxt(x_temp_path, x, delimiter=",")
        cmd = f'Rscript get_cov.R {x_temp_path} {w_temp_path} {rho} {n_his}'
        print('Estimating covariance for the {}th-{}th x in x_train...'.format(start, end))
        os.system(cmd)
        cov = load_matrix(w_temp_path)
        cov = np.asarray(cov).squeeze()
        np.savetxt(f, cov, delimiter=",")

    os.remove(x_temp_path)
    os.remove(w_temp_path)
    f.close()


if __name__ == '__main__':
    get_covariance()
