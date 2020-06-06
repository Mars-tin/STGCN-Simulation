import os
import argparse
import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stgcn import STGCN
from GaussianCopula import CopulaLoss
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, get_covariance


use_gpu = True
visualized = False
dataset = "METR-LA"
loss_fn = "copula"
error = 1e-3
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 500
patience = 30
batch_size = 50

tau_list = [0.01, 0.1, 1]
gamma_list = [0.1, 1]
seed_list = range(3)

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args.device)


def train_epoch(training_input, training_target, batch_size, sigma):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)

        if hasattr(loss_criterion, 'requires_cov'):
            loss = loss_criterion(out, y_batch, sigma)
        else:
            loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':

    if dataset == "METR-LA":
        A, X, means, stds = load_metr_la_data()
    else:
        print("Unknown dataset.")
        exit(0)

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave_cpu = get_normalized_adj(A)

    df = {"tau": [], "gamma": [], "seed": [], "MSE": [], "MAE": []}
    if loss_fn == "mse":
        tau_list = [0.1]
        gamma_list = [0.1]

    for tau in tau_list:
        for gamma in gamma_list:
            for seed in seed_list:

                df["tau"].append(tau)
                df["gamma"].append(gamma)
                df["seed"].append(seed)

                sigma = get_covariance(A_wave_cpu, tau, gamma)
                torch.manual_seed(seed)
                A_wave = torch.from_numpy(A_wave_cpu)
                A_wave = A_wave.to(device=args.device)

                net = STGCN(A_wave.shape[0],
                            training_input.shape[3],
                            num_timesteps_input,
                            num_timesteps_output).to(device=args.device)

                optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

                if loss_fn == "mse":
                    loss_criterion = nn.MSELoss()
                elif loss_fn == "copula":
                    loss_criterion = CopulaLoss()
                else:
                    print("Unknown loss_fn.")
                    exit(0)

                training_losses = []
                validation_losses = []
                validation_maes = []

                if visualized:
                    plt.ion()
                    fig, axes = plt.subplots(1, figsize=(10, 5))
                    plt.suptitle('STGCN Training')
                    axes.set_xlabel('Epoch')
                    axes.set_ylabel('Loss')

                rem = patience
                idx = 0
                best_loss = float('inf')

                for epoch in range(epochs):
                    # Early stop
                    if patience < 0:
                        break

                    loss = train_epoch(training_input, training_target,
                                       batch_size=batch_size, sigma=sigma)
                    training_losses.append(loss)

                    # Run validation
                    with torch.no_grad():
                        net.eval()
                        val_input = val_input.to(device=args.device)
                        val_target = val_target.to(device=args.device)
                        permutation = torch.randperm(training_input.shape[0])

                        val_loss_batch = []
                        mae_batch = []
                        for i in range(0, training_input.shape[0], batch_size):
                            indices = permutation[i:i + batch_size]
                            X_batch, y_batch = training_input[indices], training_target[indices]
                            X_batch = X_batch.to(device=args.device)
                            y_batch = y_batch.to(device=args.device)

                            out = net(A_wave, X_batch)
                            if hasattr(loss_criterion, 'requires_cov'):
                                val_loss = loss_criterion(out, y_batch, sigma).to(device="cpu")
                            else:
                                val_loss = loss_criterion(out, y_batch).to(device="cpu")

                            val_loss_batch.append(val_loss.detach().numpy().item())

                            out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
                            target_unnormalized = y_batch.detach().cpu().numpy()*stds[0]+means[0]
                            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                            mae_batch.append(mae)

                        validation_losses.append(np.mean(val_loss_batch))
                        validation_maes.append(np.mean(mae_batch))

                        out = None
                        val_input = val_input.to(device="cpu")
                        val_target = val_target.to(device="cpu")

                    print("\nEpoch: {}".format(epoch))
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))
                    print("Validation MAE: {}".format(validation_maes[-1]))

                    checkpoint_path = "checkpoints/"
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    with open("checkpoints/losses.pk", "wb") as fd:
                        pk.dump((training_losses, validation_losses, validation_maes), fd)

                    valid_loss = validation_losses[-1]
                    if valid_loss < best_loss - error:
                        rem = patience
                        best_loss = valid_loss
                        idx = epoch

                    rem -= 1

                mse_fn = validation_losses[idx]
                mae_fn = validation_maes[idx]
                df["MSE"].append(mse_fn)
                df["MAE"].append(mae_fn)
                print("\nThe MSE loss on test dataset is:", mse_fn)
                print("The MAE on test dataset is:", mae_fn)
                print("This is obtained in epoch", idx)

    pd_writer = pd.DataFrame(df)
    pd_writer.to_csv('output.csv', index=False, header=False)

    if visualized:
        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.plot(validation_maes, label="validation MAE")
        plt.legend()
        fig.savefig("STGCN_training_{}".format(seed), dpi=200)
        plt.show()
