"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

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

