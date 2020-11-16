"""
This module implements a bidirectional LSTM in PyTorch.
You should fill in code into indicated sections.
Date: 2020-11-09
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class biLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(biLSTM, self).__init__()
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


class LSTMCell(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTMCell, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x, c, h):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
