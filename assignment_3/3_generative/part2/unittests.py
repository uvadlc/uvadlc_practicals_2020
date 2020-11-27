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

import unittest
import numpy as np
import torch

from models import GeneratorMLP, DiscriminatorMLP


class TestGenerator(unittest.TestCase):

    @torch.no_grad()
    def test_shape(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 64
        gen = GeneratorMLP(z_dim=z_dim, output_shape=[1, 28, 28])
        z = torch.randn(4, z_dim)
        imgs = gen(z)
        self.assertTrue(len(imgs.shape) == 4 and
                        all([imgs.shape[i] == o for i,
                             o in enumerate([4, 1, 28, 28])]),
                        msg="The output of the generator should be an image with shape [B,C,H,W].")

    @torch.no_grad()
    def test_output_values(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 20
        gen = GeneratorMLP(z_dim=z_dim, hidden_dims=[64], 
                           output_shape=[1, 28, 28])
        z = torch.randn(128, z_dim) * 50
        imgs = gen(z)
        self.assertTrue((imgs >= -1).all() and (imgs <= 1).all(),
                        msg="The output of the generator should have values between -1 and 1. " + \
                            "A tanh as output activation function might be missing.")
        self.assertTrue((imgs < 0).any(),
                        msg="The output of the generator should have values between -1 and 1, " + \
                            "but seems to be missing negative values in your model.")


class TestDiscriminator(unittest.TestCase):

    @torch.no_grad()
    def test_output_values(self):
        np.random.seed(42)
        torch.manual_seed(42)
        disc = DiscriminatorMLP(input_dims=784)
        z = torch.randn(128, 784)
        preds = disc(z)
        self.assertTrue((preds < 0).any(),
                        msg="The output of the discriminator does not have any negative values. " +
                            "You might be applying a sigmoid on the discriminator output. " + \
                            "It is recommended to work on logits instead as this is numercially more stable. " + \
                            "Ensure that you are using the correct loss accordings (BCEWithLogits instead of BCE).")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerator)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiscriminator)
    unittest.TextTestRunner(verbosity=2).run(suite)
