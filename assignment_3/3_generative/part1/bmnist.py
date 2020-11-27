################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
import numpy as np


class Binarize:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        return (img > self.threshold).float()


def bmnist(root='../data/', batch_size=128, num_workers=4, download=True):
    """
    Returns data loaders for a binarized version of the MNIST dataset.
    Note that the original "binary" MNIST dataset was created by sampling
    each pixel based on the continuous value between 0 and 1. For simplicity
    and easier reproducibility, we simply take a threshold here.

    Inputs:
        root - Directory in which the MNIST dataset should be downloaded. It is better to
               use the same directory as the part2 of the assignment to prevent duplicate
               downloads.
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
        download - If True, MNIST is downloaded if it cannot be found in the specified
                   root directory.
    """
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          Binarize(threshold=0.5)])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=data_transforms, download=download)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, download=download)

    train_dataset = data.dataset.Subset(dataset, np.arange(45000))
    val_dataset = data.dataset.Subset(dataset, np.arange(45000, 50000))

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True)
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)

    return train_loader, val_loader, test_loader
