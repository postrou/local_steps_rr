import unittest
from itertools import product

import numpy as np

from src.optimizers_torch.shuffle_once_sampler import ShuffleOnceSampler


class ShufflingTorchTest(unittest.TestCase):
    def test_shuffling(self):
        data = np.arange(10)
        sampler = ShuffleOnceSampler(data)
        
        n_epochs = 3
        indices = []
        for i_epoch in range(n_epochs):
            sampler_it = iter(sampler)
            idx = []
            for i in range(len(data)):
                el = next(sampler_it)
                idx.append(el)
            indices.append(idx)
            self.assertEqual(len(idx), len(data))

        for idx_1, idx_2 in product(indices, indices):
            if idx_1 is not idx_2:
                self.assertNotEqual(idx_1, idx_2)
                self.assertEqual(set(idx_1), set(idx_2))
