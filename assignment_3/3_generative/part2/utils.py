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

from statistics import mean
from collections import defaultdict


class TensorBoardLogger(object):

    def __init__(self, summary_writer, avg_window=50, name=None):
        """
        Class that summarizes some logging code for TensorBoard.
        Specifically, we average the last "avg_window" values for each
        log entry before adding it to TensorBoard. Reduces the number of 
        points in TensorBoard and the disk space.

        Inputs:
                summary_writer - Summary Writer object from torch.utils.tensorboard
                avg_window - Number of values to average before adding to TensorBoard
                name - Tab name in TensorBoard's scalars
        """
        self.summary_writer = summary_writer
        self.avg_window = avg_window
        if name is None:
            self.name = ""
        else:
            self.name = name + "/"

        self.value_dict = defaultdict(lambda: 0)
        self.steps = defaultdict(lambda: 0)
        self.global_step = 0

    def add_values(self, log_dict):
        """
        Function for adding a dictionary of logging values to this logger. Note that
        this function increases the global_step, and hence should be called once per
        training iteration/validation epoch.

        Inputs:
                log_dict - Dictionary of string to Tensor with the values to plot.
        """

        self.global_step += 1

        for key, val in log_dict.items():
            # Add new value to averages
            val = val.detach().cpu()
            self.value_dict[key] += val
            self.steps[key] += 1
            # Plot to TensorBoard every avg_window steps
            if self.steps[key] >= self.avg_window:
                avg_val = self.value_dict[key] / self.steps[key]
                self.summary_writer.add_scalar(self.name + key,
                                               avg_val,
                                               global_step=self.global_step)
                self.value_dict[key] = 0
                self.steps[key] = 0
