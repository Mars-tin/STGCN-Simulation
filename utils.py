import torch
import numpy as np


def slice_covariance(cov, resolution, batch_size, iter):
    idx = iter * batch_size // resolution
    sigma = cov[idx]
    n = sigma.shape[0]
    lower = iter * batch_size
    upper = min(n, lower + batch_size)
    mask = torch.zeros(n).to(dtype=torch.bool)
    mask[lower:upper] = True
    return sigma[mask].t()[mask].t()


"""
def get_adjecency_matrix(weight, feature, device, density=30):
    n = feature.shape[0]
    m = density * n

    X = torch.squeeze(feature)            # (n, h, d0)
    W = torch.Tensor(weight).to(device)   # (d0, d1)
    prod = X.matmul(W)  # (n, h, d1)
    logits = torch.zeros(n, n)
    for i in range(n):
        print(i)
        for j in range(i + 1, n):
            feat = prod[i] - prod[j]
            logits[i][j] = -torch.norm(feat)
            logits[j][i] = logits[i][j]
    threshold = torch.sort(torch.reshape(logits, (-1,)))[-m]
    adj = (logits >= threshold).astype(float)
    return adj
"""


def get_adjecency_matrix(weight, feature, density=30):
    """
    weight(W): d0 * d1 (228 * 228)
    feature(X): n * 1 * h * d0 (9316 * 1 * 12 * 228)
    adj(A): n * n
    """
    n = feature.shape[0]
    m = density * n

    X = np.asarray(feature.to(device="cpu")).squeeze(axis=1)    # (n, h, d0)
    prod = X.dot(weight)                                        # (n, h, d1)
    logits = np.zeros((n, n))
    for i in range(n):
        if i % 200 == 0:
            print(i)
        for j in range(i + 1, n):
            feat = prod[i] - prod[j]
            logits[i][j] = -np.linalg.norm(feat)
            logits[j][i] = logits[i][j]
    threshold = np.sort(logits.reshape(-1))[-m]
    adj = (logits >= threshold).astype(float)

    return adj


def scaled_laplacian(A):
    """
    Calculate the rescaled laplacian given the weight matrix.
    (Fomula above eq(3))
    """
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
    """
    Compute the chebyshev polynomial kernel
    """
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def evaluate_model(model, loss_fn, data_iter, cov, resolution, batch_size):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        iter = 0
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            if hasattr(loss_fn, 'requires_cov'):
                sigma = slice_covariance(cov, resolution, batch_size, iter)
                iter += 1
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
