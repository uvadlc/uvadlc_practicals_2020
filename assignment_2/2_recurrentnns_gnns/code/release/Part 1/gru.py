"""
This module implements a GRU in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(GRU, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = device

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################


    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
