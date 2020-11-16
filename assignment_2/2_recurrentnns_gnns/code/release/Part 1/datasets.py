###############################################################################
# MIT License
#
# Copyright (c) 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall/Winter 2020
# Date Created: 2020-11-05
###############################################################################


import random
import numpy as np

import torch
import torch.utils.data as data

import sys
import math

'''
Helper functions
'''


def check_even(length):
    '''
    check whether consecutive zeros are an even number
    if so return True
    else return False
    '''
    if length % 2 == 0:
        return True
    return False


def check_baum_sweet(binary_string):
    zeros_list = binary_string.split('1')
    lengths = [check_even(len(block)) for block in zeros_list]

    # if list contains false, no uneven numbers return 1, otherwise 0
    if False not in lengths:
        return 1
    else:
        return 0


def gen_baum_sweet(max_len):
    '''
    Generate a Baum-Sweet sequence.
    https://en.wikipedia.org/wiki/Baum%E2%80%93Sweet_sequence
    Target is 1 if binary representation of sampled number contains
    no odd blocks of zeros, otherwise 0.
    conditional on x > 0
    '''
    # 0 is an exception to this rule
    stop_sampling = False
    while not stop_sampling:
        sampled_integer = random.randint(10, 10**max_len)

        binary_string = bin(sampled_integer).lstrip('0').lstrip('b')

        baum_sweet_label = check_baum_sweet(binary_string)

        if baum_sweet_label == 1:
            break

        if (random.random() < 0.016724):
            break

    return binary_string, baum_sweet_label


def encode_X(max_len, binary_string):
    length_string = len(binary_string)
    prepend = max_len - length_string
    datapoint = np.zeros((2, max_len))

    for x in range(0, length_string):
        datapoint[int(binary_string[x]), prepend + x] = 1

    return datapoint, prepend, length_string


'''
DataLoaders
'''


class BaumSweetSequenceDataset(data.Dataset):
    '''
    Generates the baum sweet seqeuence for a digit with a maximum length
    the returned input has length seq_length * 4 - 1

    '''

    def __init__(self, seq_length):
        self.seq_length = seq_length  # maximum length
        self.bin_seq_length = 4*self.seq_length


    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        # one hot encoded baum sweet sequence and label

        encoded_sequence, label = self.generate_baum_sweet()
        return encoded_sequence, label

    def generate_baum_sweet(self):
        # converts a number into a binary sequence and calculates the
        # baum_sweet number

        # predefined parameters
        max_len = self.bin_seq_length
        result_mat = np.zeros((3, max_len))
        label = np.zeros(1)

        # sample an integer, convert and run baum
        binary_string, baum_sweet = gen_baum_sweet(self.seq_length)
        encoding, prepend, length_string = encode_X(max_len, binary_string)

        # concatenate
        none_values = np.concatenate([np.ones((1, prepend)), np.zeros((1,
                                     length_string))], 1)
        result_mat = np.concatenate([none_values, encoding], 0)
        label[0] = int(baum_sweet)

        # undo one hot encoding
        result_mat = np.argmax(result_mat, 0)

        return torch.FloatTensor(np.expand_dims(result_mat, 1)).permute(1,0), int(label)


class BinaryPalindromeDataset(data.Dataset):

    def __init__(self, seq_length):

        # sequence length in this case will increase by 4 for each increment
        self.seq_length = seq_length

    def __len__(self):

        return sys.maxsize

    def __getitem__(self, idx):
        # Keep last digit as target label. Note: one-hot encoding for inputs is
        # more suitable for training, but this also works.
        full_palindrome = self.generate_binary_palindrome()
        # Split palindrome into inputs (N-1 digits) and target (1 digit)

        return torch.FloatTensor(np.expand_dims(full_palindrome[0:-1] + 1, 1)), int(full_palindrome[-1])

    def generate_binary_palindrome(self):
        # Generates a single, random palindrome number of 'length' digits.
        seq_length = self.seq_length
        bin_seq_length = seq_length * 4 + 2

        # generate a sequence of numbers, binarize to obtain string of 1 and 1
        left = [np.random.randint(0, 9)
                for _ in range(math.ceil(seq_length / 2))]

        binary_left = [[np.float32(y)
                        for y in bin(x).lstrip('0').lstrip('b')] for x in left]

        unpacked_left = []
        for x in binary_left:
            unpacked_left.extend(x)

        # to avoid only having 1's at the end and beginning
        # flip the string flip left sometimes
        np_left = np.array(unpacked_left)

        if 0.5 < random.random():
            np_left = np.flip(np_left)

        np_right = np.flip(np_left, 0) if bin_seq_length % 2 == 0 else np.flip(
                                                                np_left[:-1],
                                                                0)

        binary_palindrome = np.concatenate((np_left, np_right))

        padding = bin_seq_length - len(binary_palindrome)
        padded_palindrome = binary_palindrome
        if padding > 0:
            padded_palindrome = np.concatenate((-np.ones(padding),
                                                binary_palindrome))

        return padded_palindrome.astype(np.double)


class RandomCombinationsDataset(data.Dataset):
    """
    The sequence will consist of N integer numbers which are sampled at
    random without replacement from the set of integers [0, N]. Here N
    represents the sequence length. The example below defines an example
    of a training (input, label) pair.
    Example:
        Input: 0,1,2,3,4,6,7,8,9
        Label: 5
    """
    def __init__(self, seq_length):
        # sequence length in this case will increase by 4 for each increment
        self.seq_length = seq_length

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        seq = np.random.permutation(self.seq_length)
        return torch.FloatTensor(seq[:-1]), seq[-1]
