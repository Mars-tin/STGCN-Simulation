# STGCN Simulation
## Intro

PyTorch simulation of the spatio-temporal graph convolutional network proposed in [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875) by Bing Yu, Haoteng Yin, Zhanxing Zhu.

The `pytorch` implementation is modified upon from [STGCN-PyTorch](https://github.com/Aguin/STGCN-PyTorch).

The traffic data source is obtained from http://pems.dot.ca.gov/.

## Results

Results reported in the original paper:

| pred(min) | MAE  | MAPE  | RMSE |
| --------- | ---- | ----- | ---- |
| 15        | 2.25 | 5.26% | 4.04 |
| 30        | 3.03 | 7.33% | 5.7  |
| 45        | 3.57 | 8.69% | 6.77 |

Results simulated:

| pred(min) | seed | shuffled | MAE  | MAPE  | RMSE |
| --------- | ---- | -------- | ---- | ----- | ---- |
| 15        | 0    | 0        | 2.23 | 5.17% | 3.98 |
| 15        | 0    | 1        | 2.26 | 5.23% | 4.00 |
| 15        | 1    | 0        | 2.23 | 5.18% | 4.00 |
| 15        | 1    | 1        | 2.22 | 5.21% | 3.99 |
| 15        | 2    | 0        | 2.24 | 5.17% | 3.99 |
| 15        | 2    | 1        | 2.25 | 5.30% | 4.00 |
| 30        | 0    | 0        | 2.98 | 7.21% | 5.36 |
| 30        | 0    | 1        | 2.9  | 7.12% | 5.34 |
| 30        | 1    | 0        | 2.98 | 7.21% | 5.36 |
| 30        | 1    | 1        | 2.98 | 7.40% | 5.35 |
| 30        | 2    | 0        | 2.96 | 7.17% | 5.33 |
| 30        | 2    | 1        | 2.97 | 7.09% | 5.35 |
| 45        | 0    | 0        | 3.41 | 8.35% | 6.06 |
| 45        | 0    | 1        | 3.32 | 8.18% | 5.99 |
| 45        | 1    | 0        | 3.37 | 8.27% | 6.04 |
| 45        | 1    | 1        | 3.34 | 8.50% | 6.11 |
| 45        | 2    | 0        | 3.37 | 8.32% | 6.04 |
| 45        | 2    | 1        | 3.35 | 8.46% | 6.10 |



