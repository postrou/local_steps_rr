import numpy as np

from .shuffling import Shuffling


class NastyaL0L1(Shuffling):

    def __init__(
        self,
        c_1,
        c_2,
        inner_step_size,
        steps_per_permutation=None,
        batch_size=1,
        *args,
        **kwargs
    ):
        super().__init__(
            steps_per_permutation=steps_per_permutation,
            batch_size=batch_size,
            *args,
            **kwargs
        )
        self.c_1 = c_1
        self.c_2 = c_2
        self.inner_step_size = inner_step_size
        self.outer_step_size = None
        self.g = None

    def step(self):
        if self.g is None:
            self.g = np.zeros_like(self.x)
        idx, normalization = self.permute()
        self.i += self.batch_size
        self.grad = self.loss.stochastic_gradient(
            self.x, 
            idx=idx, 
            normalization=normalization
        )

        if self.i < self.loss.n:
            self.x -= self.inner_step_size * self.grad
            self.g += self.grad
        else:
            # grad_start_epoch = self.loss.stochastic_gradient(
            #     self.x_start_epoch, 
            #     idx=range(self.loss.n),
            #     normalization=self.loss.n
            # )
            # self.outer_step_size = 1 / (self.c_1 + self.c_2 * self.loss.norm(grad_start_epoch))
            self.g *= self.batch_size / self.loss.n
            self.outer_step_size = 1 / (self.c_1 + self.c_2 * self.loss.norm(self.g))
            self.x -= self.outer_step_size *  self.g
            self.x_start_epoch = self.x.copy()
            self.g = np.zeros_like(self.x)
        
        self.i %= self.loss.n
