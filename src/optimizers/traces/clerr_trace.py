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