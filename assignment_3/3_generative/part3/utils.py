"""
This file contains utilities needed for visualizing densities and generating
mesh grids of datapoints.
Adapted from: https://dfdazac.github.io/02-flows.html

Author: Deep Learning Course, C.Winkler | Fall 2020
Date Created: 2020-11-25
"""

import matplotlib.pyplot as plt
import torch


def make_mesh():
    domain = torch.linspace(-2.5, 2.5, 50)
    z1, z2 = torch.meshgrid((domain, domain))
    z = torch.stack((z1, z2), dim=-1)
    return z1, z2, z


def plot_contours(p_z, extent=(-2.5, 2.5, 2.5, -2.5)):
    cs = plt.contourf(p_z, extent=extent, cmap='Blues')
    cs.changed()


def plot_density(distribution):
    # Create evenly spaced data points from the input domain
    z1, z2, z = make_mesh()
    # evaluate p_u at all points u
    p_z = distribution.log_prob(z).exp()
    plot_contours(p_z)
