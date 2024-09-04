import unittest

import torch
from src.optimizers_torch import NASTYA


class NASTYATest(unittest.TestCase):
    def test_inner_step(self):
        params = torch.tensor([1., 2., 3.], requires_grad=True)
        params_start = params.clone().detach()
        optimizer = NASTYA(params=[params], n_batches=1, lr=1, outer_lr=0)
        result = params[0] * 2 + params[1] ** 2 + 2 * params[2] ** 2
        result.backward()
        optimizer.step()
        self.assertListEqual(params.tolist(), [-1., -2., -9.])
        optimizer.outer_step([params_start])
        self.assertListEqual(params.tolist(), [1., 2., 3.])
        
    def test_outer_step(self):
        params = torch.tensor([1., 2., 3.], requires_grad=True)
        params_start = params.clone().detach()
        optimizer = NASTYA(params=[params], n_batches=1, lr=0, outer_lr=1)
        result = params[0] * 2 + params[1] ** 2 + 2 * params[2] ** 2
        result.backward()
        optimizer.step()
        self.assertListEqual(params.tolist(), [1., 2., 3.])
        optimizer.outer_step([params_start])
        self.assertListEqual(params.tolist(), [-1., -2., -9.])
        