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
import torch.nn as nn

from utils import sample_reparameterize, KLD, elbo_to_bpd
from mlp_encoder_decoder import MLPEncoder, MLPDecoder
from cnn_encoder_decoder import CNNEncoder, CNNDecoder


class TestKLD(unittest.TestCase):

    @torch.no_grad()
    def test_normal_dist(self):
        mean = torch.zeros(1,1)
        log_std = torch.zeros(1,1)
        out = KLD(mean, log_std).numpy()
        self.assertTrue((out == 0).all(),
                         msg="The KLD for a normal distribution with mean 0 and std 1 must be 0, but is %s" % (str(out[0])))

    @torch.no_grad()
    def test_symmetry(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(5):
            mean = torch.randn(16,4)
            log_std = torch.randn(16,4)
            out1 = KLD(mean, log_std).numpy()
            out2 = KLD(-mean, log_std).numpy()
            self.assertTrue((out1 == out2).all(),
                            msg="The KLD must be symmetric for the mean values.\n"+\
                                "Positive mean:%s\nNegative mean:%s" % (str(out1), str(out2)))


class TestReparameterization(unittest.TestCase):

    def test_gradients(self):
        np.random.seed(42)
        torch.manual_seed(42)
        mean = torch.randn(16,4, requires_grad=True)
        log_std = torch.randn(16,4, requires_grad=True)
        out = sample_reparameterize(mean, log_std.exp())
        try:
            out.sum().backward()
        except RuntimeError:
            assert False, "The output tensor of reparameterization does not include the mean and std tensor in the computation graph."
        self.assertTrue(mean.grad is not None,
                         msg="Gradients of the mean tensor are None")
        self.assertTrue(log_std.grad is not None,
                         msg="Gradients of the standard deviation tensor are None")

    @torch.no_grad()
    def test_distribution(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(10):
            mean = torch.randn(1,)
            std = torch.randn(1,).exp()
            mean, std = mean.expand(20000,), std.expand(20000,)
            out = sample_reparameterize(mean, std)
            out_mean = out.mean()
            out_std = out.std()
            self.assertLess((out_mean - mean[0]).abs(), 1e-1,
                            msg="Sampled distribution does not match the mean.")
            self.assertLess((out_std - std[0]).abs(), 1e-1,
                            msg="Sampled distribution does not match the standard deviation.")


class TestBPD(unittest.TestCase):

    @torch.no_grad()
    def test_random_image_shape(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(10):
            img = torch.zeros(8, *[np.random.randint(10,20) for _ in range(3)]) + 0.5
            elbo = -np.log(img).sum(dim=(1,2,3))
            bpd = elbo_to_bpd(elbo, img.shape)
            self.assertTrue(((bpd - 1.0).abs() < 1e-5).all(),
                             msg="The bits per dimension score for a random image has to be 1. Given scores: %s" % str(bpd))


class TestMLPEncoderDecoder(unittest.TestCase):

    @torch.no_grad()
    def test_encoder(self):
        np.random.seed(42)
        torch.manual_seed(42)
        all_means, all_log_std = [], []
        for test_num in range(10):
            z_dim = np.random.randint(2,40)
            encoder = MLPEncoder(input_dim=784, z_dim=z_dim)
            img = torch.randn(32, 1, 28, 28)
            mean, log_std = encoder(img)
            self.assertTrue((mean.shape[0] == 32 and mean.shape[1] == z_dim),
                             msg="The shape of the mean output should be batch_size x z_dim")
            self.assertTrue((log_std.shape[0] == 32 and log_std.shape[1] == z_dim),
                             msg="The shape of the log_std output should be batch_size x z_dim")
            all_means.append(mean.reshape(-1))
            all_log_std.append(log_std.reshape(-1))
        means = torch.cat(all_means, dim=0)
        log_std = torch.cat(all_log_std, dim=0)
        self.assertTrue((means > 0).any() and (means < 0).any(), msg="Only positive or only negative means detected. Are you sure this is what you want?")
        self.assertTrue((log_std > 0).any() and (log_std < 0).any(), msg="Only positive or only negative means detected. Are you sure this is what you want?")

    @torch.no_grad()
    def test_decoder(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 20
        decoder  = MLPDecoder(z_dim=20, output_shape=[1,28,28])
        z = torch.randn(64, z_dim)
        imgs = decoder(z)
        self.assertTrue(len(imgs.shape) == 4 and all([imgs.shape[i] == o for i,o in enumerate([64,1,28,28])]),
                         msg="Output of the decoder should be an image with shape [B,C,H,W], but got: %s." % str(imgs.shape))
        self.assertTrue((imgs < 0).any(),
                         msg="The output of the decoder does not have any negative values. " + \
                             "You might be applying a sigmoid on the decoder output. " + \
                             "It is recommended to work on logits instead as this is numercially more stable. " + \
                             "Ensure that you are using the correct loss accordings (BCEWithLogits instead of BCE).")


class TestCNNEncoderDecoder(unittest.TestCase):

    @torch.no_grad()
    def test_encoder(self):
        np.random.seed(42)
        torch.manual_seed(42)
        skip_test = False
        try:
            enc = CNNEncoder()
        except NotImplementedError:
            skip_test = True

        if not skip_test:
            all_means, all_log_std = [], []
            for test_num in range(10):
                z_dim = np.random.randint(2,40)
                encoder = CNNEncoder(z_dim=z_dim)
                img = torch.randn(32, 1, 28, 28)
                mean, log_std = encoder(img)
                self.assertTrue((mean.shape[0] == 32 and mean.shape[1] == z_dim),
                                 msg="The shape of the mean output should be batch_size x z_dim")
                self.assertTrue((log_std.shape[0] == 32 and log_std.shape[1] == z_dim),
                                 msg="The shape of the log_std output should be batch_size x z_dim")
                all_means.append(mean.reshape(-1))
                all_log_std.append(log_std.reshape(-1))
            means = torch.cat(all_means, dim=0)
            log_std = torch.cat(all_log_std, dim=0)
            self.assertTrue((means > 0).any() and (means < 0).any(), msg="Only positive or only negative means detected. Are you sure this is what you want?")
            self.assertTrue((log_std > 0).any() and (log_std < 0).any(), msg="Only positive or only negative means detected. Are you sure this is what you want?")

    @torch.no_grad()
    def test_decoder(self):
        np.random.seed(42)
        torch.manual_seed(42)
        skip_test = False
        try:
            enc = CNNDecoder()
        except NotImplementedError:
            skip_test = True

        if not skip_test:
            z_dim = 20
            decoder  = CNNDecoder(z_dim=20)
            z = torch.randn(64, z_dim)
            imgs = decoder(z)
            self.assertTrue(len(imgs.shape) == 4 and all([imgs.shape[i] == o for i,o in enumerate([64,1,28,28])]),
                             msg="Output of the decoder should be an image with shape [B,C,H,W], but got: %s." % str(imgs.shape))
            self.assertTrue((imgs < 0).any(),
                             msg="The output of the decoder does not have any negative values. " + \
                                 "You might be applying a sigmoid on the decoder output. " + \
                                 "It is recommended to work on logits instead as this is numercially more stable. " + \
                                 "Ensure that you are using the correct loss accordings (BCEWithLogits instead of BCE).")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKLD)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestReparameterization)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBPD)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLPEncoderDecoder)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCNNEncoderDecoder)
    unittest.TextTestRunner(verbosity=2).run(suite)

