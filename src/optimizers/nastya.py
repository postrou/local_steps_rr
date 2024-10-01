import numpy as np

from .shuffling import Shuffling
from .traces import ClERRTrace


class NASTYA(Shuffling):
    def __init__(
        self,
        inner_step_size,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trace = ClERRTrace(self.loss)
        self.inner_step_size = inner_step_size

    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.x_start_epoch = self.x.copy()
        self.i = 0
     
    def step(self):
        if self.i == 0:
            self.grad_estimator = np.zeros_like(self.x)

        idx, _ = self.permute()
        self.stoch_grad = self.loss.stochastic_gradient(
            self.x, 
            idx=idx, 
            normalization=None
        )
        self.grad_estimator += self.stoch_grad * (len(idx) / self.loss.n)

        self.perform_inner_step()
        self.i += self.batch_size
        if self.i >= self.loss.n:
            self.perform_outer_step()
            self.i = 0

    def permute(self):
        if self.it == 0:
            # it enters here during 0-th step, so self.permutation is always
            #   initialized
            self.permutation = np.random.permutation(self.loss.n)
            self.sampled_permutations += 1

        idx_perm = np.arange(self.i, min(self.loss.n, self.i + self.batch_size))
        idx = self.permutation[idx_perm]
        return idx, None

    def perform_inner_step(self):
        self.x -= self.inner_step_size * self.stoch_grad

    def perform_outer_step(self):
        self.x = self.x_start_epoch - self.lr0 * self.grad_estimator
        self.grad_estimator = np.zeros_like(self.x)
        self.x_start_epoch = self.x.copy()
