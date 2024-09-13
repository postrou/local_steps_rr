import unittest

import torch

from src.optimizers_torch import ClERRHeuristic


class ClERRHeuristicTest(unittest.TestCase):
    def test_clipping(self):
        x = torch.tensor([2.] * 4, requires_grad=True)
        c_0, c_1 = 1, 0
        optimizer = ClERRHeuristic(c_0, c_1, in_clip_level=1, params=[x], lr=1)

        result = x.norm().mul(3)
        result.backward()
        # grad = [1.5, 1.5, 1.5, 1.5]
        optimizer.step()
        # grad = [0.5, 0.5, 0.5, 0.5]
        for x_i in x:
            self.assertAlmostEqual(x_i.item(), 1.5, places=4)

    def test_g_update(self):
        x = torch.tensor([2.] * 4, requires_grad=True)
        x_start_epoch = x.detach().clone()
        c_0, c_1 = 1, 1/4
        optimizer = ClERRHeuristic(c_0, c_1, in_clip_level=1, params=[x], lr=2)

        result = x.norm().mul(3)
        result.backward()
        optimizer.step()
        for x_i in x:
            self.assertAlmostEqual(x_i.item(), 1., places=4)
        optimizer.zero_grad()

        result = x.mul(2).norm() # = ||2x|| = 3**2 * 4 = 36
        result.backward()
        optimizer.step()
        for x_i in x:
            self.assertAlmostEqual(x_i.item(), 0., places=4)
        optimizer.outer_step(x_start_epoch)
        for x_i in x:
            self.assertAlmostEqual(x_i.item(), 1., places=4)
        # f(x) = ||2x|| = \sqrt{<2x, 2x>}
        # \nabla f(x) = 1/(2 \sqrt{<2x, 2x>}) * 2 * 2x = 2x / ||2x||