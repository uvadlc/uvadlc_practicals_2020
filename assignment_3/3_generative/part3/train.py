"""
This file contains the training loop for a flow based model.
Adapted from: https://dfdazac.github.io/02-flows.html

Author: Deep Learning Course, C.Winkler | Fall 2020
Date Created: 2020-11-25
"""

import os
import datetime
import argparse
import numpy as np
import torch

# model
from model import NormalizingFlow

# dataloading
from torch.utils.data import TensorDataset, DataLoader
from distributions import BimodalGaussianDiag, MultivariateGaussianDiag
from torch.utils.tensorboard import SummaryWriter

from torch.optim import Adam

# logging tool
from tqdm import trange

# plot losses & distributions
import matplotlib.pyplot as plt
from utils import make_mesh, plot_contours, plot_density


def train_nf(model, train_loader, optimizer):
    sum_bpd = 0
    # TODO: implement training loop here ...
    for x in train_loader:
        raise NotImplementedError
    avg_bpd = None
    return avg_bpd


def main(args):

    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_writer = SummaryWriter(experiment_dir)

    # Initialize distributions:
    # Base density
    base = MultivariateGaussianDiag(mu=torch.zeros(2),
                                    sigma=torch.tensor([0.3, 0.3]),
                                    dims=2)

    # Target density
    target = BimodalGaussianDiag(mu=[1 * torch.ones(2),
                                 -1 * torch.ones(2)],
                                 sigma=[torch.tensor([0.3, 0.3])]*2,
                                 dims=2)

    # Plotting the base and target density
    plt.figure(figsize=(6, 6))
    plot_density(base)
    plt.savefig(os.path.join(experiment_dir,
                             'density_base'))
    fig = plt.figure(figsize=(6, 6))
    plot_density(target)
    plt.savefig(os.path.join(experiment_dir,
                             'density_target'))
    plt.close()

    ################################################################
    # Check for yourself - Do the plots look like what you expect ?#
    ################################################################

    # Generate dataset
    dataset = TensorDataset(target.sample(10000))

    # Build data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = NormalizingFlow(base, dim=2,
                            num_flows=args.num_flows,
                            num_hidden=args.num_hidden)

    # Set training configs and logging tool
    optimizer = Adam(model.parameters(), lr=args.lr)

    # start training
    plot_model_density(model, experiment_dir, summary_writer, epoch=0)
    plot_model_samples(model, experiment_dir, summary_writer, epoch=0)

    epoch_iterator = (trange(1, args.epochs+1, desc=f"NF")
                      if args.progress_bar else range(1, args.epochs+1))

    for epoch in epoch_iterator:
        train_bpd = train_nf(model, loader, optimizer)
        summary_writer.add_scalar("bpd", train_bpd, global_step=epoch)
        if epoch % 5 == 0:
            plot_model_density(model, experiment_dir,
                               summary_writer, epoch=epoch)
            plot_model_samples(model, experiment_dir,
                               summary_writer, epoch=epoch)
        if args.progress_bar:
            desc = f'bpd: {train_bpd:.3f}'
            epoch_iterator.set_description_str(desc)
            epoch_iterator.update()


@torch.no_grad()
def plot_model_samples(model, experiment_dir, summary_writer, epoch=0):
    # #########################################################################
    # Time to sample and visualize samples from our learned distribution!     #
    # #########################################################################
    fig = plt.figure(figsize=(6, 6))
    samples = model.sample(2000).detach().cpu()
    plt.scatter(samples[:, 0], samples[:, 1], label='Samples')
    plt.scatter([1, -1], [1, -1], label='True modes')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir,
                             f'generated_samples_{epoch}'))
    summary_writer.add_figure('generated_samples', fig, global_step=epoch)
    plt.close()


@torch.no_grad()
def plot_model_density(model, experiment_dir, summary_writer, epoch=0):
    ###########################################################################
    # We can also plot the learned density !                                  #
    ###########################################################################

    # create data point meshgrid
    x1, x2, x = make_mesh()

    # evaluate log-probability
    log_px = -model(x.reshape(-1, 2))

    # convert to probability value by negation and taking the exponential
    log_px = log_px.reshape(x.shape[:2]).detach()
    px = log_px.exp()

    # Finally, plot the learned density
    fig = plt.figure(figsize=(6, 6))
    plot_contours(px)
    plt.savefig(os.path.join(experiment_dir,
                             f'learned_density_{epoch}'))
    summary_writer.add_figure('learned_density', fig, global_step=epoch)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--num_flows', default=6, type=int,
                        help='Number of flow layers to use in the model.')
    parser.add_argument('--num_hidden', default=64, type=int,
                        help='Hidden dimensionality of the linear ' +
                             'layers within the model.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size to use in optimization')

    # Other hyperparameters
    parser.add_argument('--epochs', default=40, type=int,
                        help='Number of epochs.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--log_dir', default='NF_logs/', type=str,
                        help='Directory where the tensorboard logs ' +
                             'should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    main(args)
