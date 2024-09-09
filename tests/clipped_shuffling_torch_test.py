import unittest

import torch

from src.optimizers_torch import ClippedSGD, ShuffleOnceSampler


class ClippedSGDTest(unittest.TestCase):
    def test_clipping(self):
        x = torch.tensor([2.] * 4, requires_grad=True)
        optimizer = ClippedSGD(clip_level=1, params=[x], lr=1)

        result = x.norm().mul(3)
        result.backward()
        optimizer.step()
        for x_i in x:
            self.assertAlmostEqual(x_i.item(), 1.5, places=4)