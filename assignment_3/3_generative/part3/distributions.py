"""
This file contains classes for a bimodal Gaussian distribution and a
multivariate Gaussian distribution with diagonal covariance matrix.

Author: Deep Learning Course, C.Winkler | Fall 2020
Date Created: 2020-11-25
"""

import numpy as np
import torch


def broadcast(x, a, b):
    """
    Broadcast shape of input tensors a and b to be able to perform element-wise
    multiplication along the last dimension of x.
    Inputs:
    x - Input tensor of shape [n, n, d].
    a - First input tensor of shape [d].
    b - Second input tensor of shape [d].

    Returns:
        Tensor of shape [1, 1, d]
    """
    return (t.view(((1,) * (len(x.shape)-1)) + x.shape[-1:]) for t in [a, b])


class BimodalGaussianDiag:
    """
    Class specifying a Bimodal Bivariate Gaussian distribution with diagonal
    covariance matrix. Contains functions to compute the log-likelihood and to
    sample from the distribution.

    Inputs:
        mu (list)    - List of tensors of shape of 1xdims. These are
                       the mean values of the distribution for each
                       random variable.
        sigma (list) - List of tensors of shape 1xdims. These are the
                       values of standard devations of each random variable.
        dims(int)    - Dimensionality of random vector.
    """
    def __init__(self, mu, sigma, dims):
        # TODO: Implement initalization
        self.p1 = None
        self.p2 = None
        self.mus = None
        self.sigmas = None
        self.dims = None
        raise NotImplementedError

    def log_prob(self, x):
        # TODO: Implement log probability computation
        logp = None
        raise NotImplementedError
        return logp

    def sample(self, num_samples):
        # TODO: Implement sampling procedure
        samples = None
        raise NotImplementedError
        return samples


class MultivariateGaussianDiag:
    """
    Class specifying a Multivariate Gaussian distribution with diagonal
    covariance matrix. Contains functions to compute the log-likelihood and
    sample from the distribution.

    Inputs:
        mu (list)    - List of tensors of shape of 1xdims. These are
                       the mean values of the distribution for each
                       random variable.
        sigma (list) - List of tensors of shape 1xdims. These are the
                       values of standard devations of each random variable.
        dims(int)    - Dimensionality of random vector.
    """
    def __init__(self, mu, sigma, dims):
        super().__init__()
        # TODO: Implement initalization
        logp = None
        raise NotImplementedError

    def log_prob(self, x):
        # TODO: Implement log probability computation
        logp = None
        raise NotImplementedError
        return log_p

    def sample(self, num_samples):
        # TODO: Implement sampling procedure
        samples = None
        raise NotImplementedError
        return samples
