import unittest
import numpy as np
import torch
import torch.nn as nn

from modules import LinearModule, SoftMaxModule, CrossEntropyModule
from modules import ELUModule
from custom_layernorm import CustomLayerNormAutograd, CustomLayerNormManualFunction, CustomLayerNormManualModule
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class TestLayerNorm(unittest.TestCase):
    
    def test_autograd(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(10):
            n_batch = int(np.random.choice(range(16, 32)))
            n_neurons = int(np.random.choice(range(64, 256)))
            x = 2 * torch.randn(n_batch, n_neurons, requires_grad=True) + 10
            bn_auto = CustomLayerNormAutograd(n_neurons)
            y_auto = bn_auto(x)
            self.assertLess(np.max(np.abs(y_auto.mean(dim=1).data.numpy())), 1e-5)
            self.assertLess(np.max(np.abs(y_auto.var(dim=1).data.numpy() - 1)), 1e-1)
    
    def test_manual_function(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(5):
            n_batch = int(np.random.choice(range(8, 32)))
            n_neurons = int(np.random.choice(range(64, 256)))
            x = 2 * torch.randn(n_batch, n_neurons, requires_grad=True) + 10
            input = x.double()
            
            # test the bias is added correctly
            gamma = torch.ones(n_neurons, dtype=torch.float64, requires_grad=True)
            beta = 100 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
            bn_manual_fct = CustomLayerNormManualFunction(n_neurons)
            y_manual_fct = bn_manual_fct.apply(input, gamma, beta)
            grad_correct = torch.autograd.gradcheck(bn_manual_fct.apply, (input, gamma, beta))
            self.assertLess(np.max(np.abs(y_manual_fct.mean(dim=1).data.numpy() - beta.mean().item())), 1e-5)
            self.assertEqual(grad_correct, True)
            
            # test gradient
            gamma = torch.sqrt(10 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True))
            beta = 100 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
            bn_manual_fct = CustomLayerNormManualFunction(n_neurons)
            grad_correct = torch.autograd.gradcheck(bn_manual_fct.apply, (input, gamma, beta))
            self.assertEqual(grad_correct, True)
    
    def test_manual_module(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(10):
            n_batch = int(np.random.choice(range(8, 32)))
            n_neurons = int(np.random.choice(range(64, 256)))
            x = 2 * torch.randn(n_batch, n_neurons, requires_grad=True) + 10
            bn_manual_mod = CustomLayerNormManualModule(n_neurons)
            y_manual_mod = bn_manual_mod(x)
            self.assertLess(np.max(np.abs(y_manual_mod.mean(dim=1).data.numpy())), 1e-5)
            self.assertLess(np.max(np.abs(y_manual_mod.var(dim=1).data.numpy() - 1)), 1e-1)


class TestLosses(unittest.TestCase):
    
    def test_crossentropy_loss(self):
        np.random.seed(42)
        rel_error_max = 1e-5
        
        for test_num in range(10):
            N = np.random.choice(range(1, 100))
            C = np.random.choice(range(1, 10))
            X = np.random.randn(N, C)
            y = np.random.randint(C, size=(N,))
            y = dense_to_one_hot(y, C)
            X = np.exp(X - np.max(X, axis=1, keepdims=True))
            X /= np.sum(X, axis=1, keepdims=True)
            
            loss = CrossEntropyModule().forward(X, y)
            grads = CrossEntropyModule().backward(X, y)
            
            f = lambda _: CrossEntropyModule().forward(X, y)
            grads_num = eval_numerical_gradient(f, X, verbose=False, h=1e-5)
            self.assertLess(rel_error(grads_num, grads), rel_error_max)


class TestLayers(unittest.TestCase):
    
    def test_linear_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-5
        
        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            C = np.random.choice(range(1, 10))
            x = np.random.randn(N, D)
            dout = np.random.randn(N, C)
            
            layer = LinearModule(D, C)
            
            out = layer.forward(x)
            dx = layer.backward(dout)
            dw = layer.grads['weight']
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
            dw_num = eval_numerical_gradient_array(lambda w: layer.forward(x), layer.params['weight'], dout)
            
            self.assertLess(rel_error(dx, dx_num), rel_error_max)
            self.assertLess(rel_error(dw, dw_num), rel_error_max)

    def test_elu_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-6
    
        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            x = np.random.randn(N, D)
            dout = np.random.randn(*x.shape)
        
            layer = ELUModule()
        
            out = layer.forward(x)
            dx = layer.backward(dout)
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
        
            self.assertLess(rel_error(dx, dx_num), rel_error_max)

    def test_softmax_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-5
        
        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            x = np.random.randn(N, D)
            dout = np.random.randn(*x.shape)
            
            layer = SoftMaxModule()
            
            out = layer.forward(x)
            dx = layer.backward(dout)
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
            
            self.assertLess(rel_error(dx, dx_num), rel_error_max)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLosses)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayerNorm)
    unittest.TextTestRunner(verbosity=3).run(suite)
