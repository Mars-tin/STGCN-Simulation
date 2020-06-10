import torch
import numpy as np


def get_covariance(adj, tau, gamma):
    if tau is None or gamma is None:
        return None
    L = np.diag(adj.sum(axis=0)) - adj
    cov = tau * np.linalg.inv(L + gamma * np.eye(adj.shape[0]))
    return torch.tensor(cov)


def get_adjecency_matrix(weight, feature, rate=30):
    n = feature.shape[0]
    m = rate * n
    d1 = weight.shape[1]

    prod = feature.dot(weight)
    logits = -np.linalg.norm(
        prod.reshape(1, n, d1) - prod.reshape(n, 1, d1), axis=2
    )
    threshold = np.sort(logits.reshape(-1))[-m]
    adj = (logits >= threshold).astype(float)
    return adj


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.max(np.linalg.eigvals(L)).real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def evaluate_model(model, loss_fn, data_iter, sigma):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            if hasattr(loss_fn, 'requires_cov'):
                l = loss_fn(y_pred, y, sigma)
            else:
                l = loss_fn(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE
