import unittest
from itertools import product

import numpy as np

from src.optimizers_torch import RandomReshufflingSampler, ShuffleOnceSampler


class RandomReshufflingTorchTest(unittest.TestCase):
    def test_random_shuffling(self):
        data = np.arange(10)
        sampler = RandomReshufflingSampler(data)
        
        n_epochs = 3
        indices = []
        for _ in range(n_epochs):
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


class ShuffleOnceTorchTest(unittest.TestCase):
    def test_shuffle_once(self):
        data = np.arange(10)
        sampler = ShuffleOnceSampler(data)
        
        n_epochs = 3
        indices = []
        for _ in range(n_epochs):
            sampler_it = iter(sampler)
            idx = []
            for i in range(len(data)):
                el = next(sampler_it)
                idx.append(el)
            indices.append(idx)
            self.assertEqual(len(idx), len(data))

        for idx_1, idx_2 in product(indices, indices):
            if idx_1 is not idx_2:
                self.assertTrue(any(idx_1 != data))
                self.assertTrue(any(idx_2 != data))
                self.assertEqual(idx_1, idx_2)
