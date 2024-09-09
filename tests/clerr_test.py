import unittest 

import torch

from src.optimizers_torch import ClERR


class ClERRTest(unittest.TestCase):
    def test_inner_step(self):
        params = torch.tensor([1., 2., 3.], requires_grad=True)
        params_start = params.clone().detach()
        norm_grad_start_epoch = params.norm()
        c_0, c_1 = 1e+10, 0
        optimizer = ClERR(params=[params], n_batches=1, lr=1, c_0=c_0, c_1=c_1)
        result = params[0] * 2 + params[1] ** 2 + 2 * params[2] ** 2
        result.backward()
        optimizer.step()
        self.assertListEqual(params.tolist(), [-1., -2., -9.])
        optimizer.outer_step([params_start], norm_grad_start_epoch)
        self.assertListEqual(params.tolist(), [1., 2., 3.])
 
    def test_outer_step(self):
        params = torch.tensor([1., 2., 3.], requires_grad=True)
        params_start = params.clone().detach()
        norm_grad_start_epoch = params.norm()
        c_0, c_1 = 1, 0
        optimizer = ClERR(params=[params], n_batches=1, lr=0, c_0=c_0, c_1=c_1)
        result = params[0] * 2 + params[1] ** 2 + 2 * params[2] ** 2
        result.backward()
        optimizer.step()
        self.assertListEqual(params.tolist(), [1., 2., 3.])
        optimizer.outer_step([params_start], norm_grad_start_epoch)
        self.assertListEqual(params.tolist(), [-1., -2., -9.])        

    def test_full_step_with_g(self):
        params = torch.tensor([1., 2., 3.], requires_grad=True)
        params_start = params.clone().detach()
        c_0, c_1 = 3.2, 1
        optimizer = ClERR(params=[params], n_batches=1, lr=0.1, c_0=c_0, c_1=c_1, use_g_in_outer_step=True)
        result = params[0] * 2 + params[1] ** 2 + 2 * params[2] ** 2
        result.backward()
        optimizer.step()
        for i, (p, p_check) in enumerate(zip(params.tolist(), [0.8, 1.6, 1.8])):
            self.assertAlmostEqual(p, p_check, places=3)
            
        optimizer.outer_step([params_start])
        for i, (p, p_check) in enumerate(zip(params.tolist(), [7/8, 7/4, 9/4])):
            self.assertAlmostEqual(p, p_check, places=3)

    def test_full_step_without_g(self):
        params = torch.tensor([1., 2., 3.], requires_grad=True)
        params_start = params.clone().detach()
        c_0, c_1 = 3.2, 1
        optimizer = ClERR(params=[params], n_batches=1, lr=0.1, c_0=c_0, c_1=c_1, use_g_in_outer_step=False)
        result = params[0] * 2 + params[1] ** 2 + 2 * params[2] ** 2
        result.backward()
        norm_grad_start_epoch = params.grad.norm()
        optimizer.step()
        for i, (p, p_check) in enumerate(zip(params.tolist(), [0.8, 1.6, 1.8])):
            self.assertAlmostEqual(p, p_check, places=3)
            
        optimizer.outer_step([params_start], norm_grad_start_epoch)
        for i, (p, p_check) in enumerate(zip(params.tolist(), [7/8, 7/4, 9/4])):
            self.assertAlmostEqual(p, p_check, places=3)