# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch.utils.data as data


class TextDataset(data.Dataset):

    def __init__(self, filename, seq_length):
        assert os.path.splitext(filename)[1] == ".txt"
        self._seq_length = seq_length
        self._data = open(filename, 'r').read()
        self._chars = sorted(list(set(self._data)))
        self._data_size, self._vocab_size = len(self._data), len(self._chars)
        print("Initialize dataset with {} characters, {} unique.".format(
            self._data_size, self._vocab_size))
        self._char_to_ix = {ch: i for i, ch in enumerate(self._chars)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._chars)}
        self._offset = 0

    def __getitem__(self, item):
        offset = np.random.randint(0, len(self._data)-self._seq_length-2)
        inputs = [self._char_to_ix[ch] for ch in self._data[offset:offset+self._seq_length]]
        targets = [self._char_to_ix[ch] for ch in self._data[offset+1:offset+self._seq_length+1]]
        return inputs, targets

    def convert_to_string(self, char_ix):
        return ''.join(self._ix_to_char[ix] for ix in char_ix)

    def __len__(self):
        return self._data_size

    @property
    def vocab_size(self):
        return self._vocab_size
