import time

import numpy as np

from src.loss_functions import safe_sparse_norm
from .shuffling import Shuffling
from .clipped_shuffling import ClippedShuffling


class NastyaL0L1(Shuffling):

    def __init__(
        self,
        c_0,
        c_1,
        inner_step_size,
        steps_per_permutation=None,
        batch_size=1,
        f_tolerance=None,
        *args,
        **kwargs
    ):
        super().__init__(
            steps_per_permutation=steps_per_permutation,
            batch_size=batch_size,
            lr0=0,
            *args,
            **kwargs
        )
        self.f_tolerance = f_tolerance
        self.c_1 = c_0
        self.c_2 = c_1
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
            self.g *= self.batch_size / self.loss.n
            self.outer_step_size = 1 / (self.c_1 + self.c_2 * self.loss.norm(self.g))
            self.x -= self.outer_step_size *  self.g
            self.x_start_epoch = self.x.copy()
            self.g = np.zeros_like(self.x)
        
        self.i %= self.loss.n

    def check_convergence(self):
        if self.f_tolerance is not None:
            f_tolerance_met = \
                self.loss.value(self.x) - self.loss.f_opt < self.f_tolerance
        else:
            return super().check_convergence()
        return super().check_convergence() or f_tolerance_met


class NastyaL0L1Clip(NastyaL0L1, ClippedShuffling):

    def __init__(
        self,
        c_0,
        c_1,
        inner_step_size,
        steps_per_permutation=None,
        batch_size=1,
        *args,
        **kwargs
    ):
        clip_level = c_0 / c_1
        super().__init__(
            clip_level=clip_level,
            steps_per_permutation=steps_per_permutation,
            batch_size=batch_size,
            lr0=0,
            *args,
            **kwargs
        )
        self.lr = 1 / (2 * c_0)
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
            self.g *= self.batch_size / self.loss.n
            self.grad_estimator = self.clip(self.g)
            self.x -= self.lr *  self.grad_estimator

            self.x_start_epoch = self.x.copy()
            self.g = np.zeros_like(self.x)
        
        self.i %= self.loss.n
