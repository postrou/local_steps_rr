import torch
import numpy as np


class ShuffleOnceSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        # Shuffle indices at the start of the epoch
        # np.random.seed(self.epoch)  # Optional: To make the shuffling deterministic per epoch
        np.random.shuffle(self.indices)
        
        # Return the shuffled indices sequentially
        # for i in range(0, self.num_samples, self.batch_size):
            # idx = self.indices[i : i + self.batch_size]
        for idx in self.indices:
            yield idx

    def __len__(self):
        return self.num_samples
