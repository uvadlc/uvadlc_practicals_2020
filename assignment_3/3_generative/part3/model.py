"""
This file contains a coupling layer and normalizing flow module.
Adapted from: https://dfdazac.github.io/02-flows.html

Author: Deep Learning Course, C.Winkler | Fall 2020
Date Created: 2020-11-25
"""

import torch
import torch.nn as nn


class NormalizingFlow(nn.Module):

    def __init__(self, base_density, dim, num_flows, num_hidden):
        """
        Normalizing Flow model using Coupling Layers (Dinh et al., 2015)
        as invertible transformation.

        Inputs:
            dim (int)       - dimensionality of input variable.
            num_flows (int) - nr of invertible transformations.
            num_hidden (int)- hidden layer size in MLP of coupling layers.
        """
        super().__init__()

        # Create checkerboard mask
        mask = torch.tensor([1, 0], dtype=torch.bool).repeat(dim // 2)

        self.dim = dim
        self.register_buffer('init_log_det', torch.zeros(1))

        self.layers = nn.ModuleList()
        for i in range(num_flows // 2):
            self.layers.append(Coupling(dim, num_hidden, ~mask))
            self.layers.append(Coupling(dim, num_hidden, mask))

        # base density we want to transform
        self.base_density = base_density

    def forward(self, x):
        """
        Forward function returning the NLL of the input points.

        Inputs:
            x - Input data points of shape [batch_size, dim]
        Outputs:
            nll - Negative log likelihood in the flow distribution
        """
        # TODO: implement all steps needed to compute the NLL
        nll = None
        raise NotImplementedError
        return nll

    def transform(self, z, inverse=False):
        """
        Function for transforming z to x, or x to z (if inverse True).

        Inputs:
            z       - Data points to transform. Shape: [batch_size, dim]
            inverse - If True, we invert all flows.
        Outputs:
            z       - Transformed data points. Shape: [batch_size, dim]
            log_det - Log determinant of transformations. Shape: [batch_size]
        """
        log_det = self.init_log_det
        if not inverse:
            # TODO: implement forward pass through the flow
            z, log_det = None, None
            raise NotImplementedError
        else:
            # TODO: implement inverse pass through the flow
            z, log_det = None, None
            raise NotImplementedError
        return z, log_det

    @torch.no_grad()
    def sample(self, num_samples):
        """
        Function for sampling new data points from p(x).

        Inputs:
            num_samples - Number of samples to return
        Outputs:
            x           - Samples of shape [num_samples, dim]
        """

        # sample from base density (Gaussian)
        z = self.base_density.sample(num_samples)

        # transform with flow
        x, _ = self.transform(z, inverse=True)
        return x


class Coupling(nn.Module):

    def __init__(self, dim, num_hidden, mask):
        """
        Coupling layer as proposed in (Dinh et al., 2015).

        Inputs:
            dim (int)            - input dimensionality.
            num_hidden (int)     - nr of hidden units in MLP predicting scale
                                   and translation parameters.
            mask (boolean tensor) - binary mask to alternate
                                    the dimensions that are used to mask the
                                    dimensions that are transformed.
        """
        super().__init__()

        # MLP outputting the scaling and translation params
        self.nn = nn.Sequential(nn.Linear(dim // 2, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, dim))

        # initialize the last coupling layer to
        # implement the identity transform
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

        self.register_buffer('mask', mask)

    def forward(self, z, log_det, inverse=False):
        """
        Applying the flow on an input.

        Inputs:
            z       - Input data points to be transformed.
                      Shape:[batch_size, dim]
            log_det - Log determinant of previous layers.
        Outputs:
            z       - Transformed data points. Shape: [batch_size, dim]
            log_det - Log determinant of previous layers *plus*
                      this layer's log-det
        """
        mask = self.mask
        neg_mask = ~mask

        # compute scale and translation for relevant inputs
        s_t = self.nn(z[:, mask])  # mask out dimensions fed to MLP
        s, t = torch.chunk(s_t, chunks=2, dim=1)  # predict scale and transl.

        if not inverse:
            # scale and translate dimensions which were not masked (for mixing)
            z[:, neg_mask] = z[:, neg_mask] * torch.exp(s) + t

            # compute the log determinant to keep track of
            # changes in the probability volume after scaling & translating
            log_det = log_det + torch.sum(s, dim=1)
        else:
            # scale and translate dimensions which were not masked (for mixing)
            z[:, neg_mask] = (z[:, neg_mask] - t) * torch.exp(-s)

            # compute the log determinant to keep track of
            # changes in the probability volume after scaling & translating
            log_det = log_det - torch.sum(s, dim=1)

        return z, log_det
