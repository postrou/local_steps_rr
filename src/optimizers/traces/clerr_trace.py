import numpy as np

from .stochastic_trace import StochasticTrace


class ClERRTrace(StochasticTrace):
    def __init__(self, loss):
        super().__init__(loss)
        self.inner_step_size = None
        self.outer_step_sizes_all = {}
        
    def init_seed(self):
        super().init_seed()
        self.outer_step_sizes = []
        
    def append_seed_results(self, seed):
        super().append_seed_results(seed) 
        self.outer_step_sizes_all[seed] = self.outer_step_sizes.copy()

    def convert_its_to_epochs(self, batch_size=1):
        its_per_epoch = np.ceil(self.loss.n / batch_size)
        if self.its_converted_to_epochs:
            return
        for seed, its in self.its_all.items():
            self.its_all[seed] = np.asarray(its) / its_per_epoch
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
 