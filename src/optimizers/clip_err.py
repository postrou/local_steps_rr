import numpy as np

from .shuffling import Shuffling


class ClipERR(Shuffling):

    def __init__(
        self,
        c_0,
        c_1,
        inner_step_size,
        steps_per_permutation=None,
        batch_size=1,
        f_tolerance=None,
        use_g_in_outer_step=False,
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
        self.c_1 = c_0
        self.c_2 = c_1
        self.inner_step_size = inner_step_size
        self.f_tolerance = f_tolerance
        self.use_g_in_outer_step = use_g_in_outer_step
        self.g = None
    
    def step(self):
        if self.g is None:
            self.g = np.zeros_like(self.x)
            if not self.use_g_in_outer_step:
                grad_start_epoch = self.loss.stochastic_gradient(
                    self.x,
                    idx=range(self.loss.n)
                )
                self.norm_grad_start_epoch = self.loss.norm(grad_start_epoch)

        idx, normalization = self.permute()
        self.i += self.batch_size
        self.grad = self.loss.stochastic_gradient(
            self.x, 
            idx=idx, 
            normalization=normalization
        )

        if self.i < self.loss.n:
            self.perform_inner_step()
        else:
            self.perform_outer_step()
        
        self.i %= self.loss.n

    def perform_inner_step(self):
        self.x -= self.inner_step_size * self.grad
        self.g += self.grad

    def perform_outer_step(self):
        self.g *= self.batch_size / self.loss.n

        outer_step_size = self.calculate_outer_step_size()

        self.x -= outer_step_size * self.g
        if not self.use_g_in_outer_step:
            grad_start_epoch = self.loss.stochastic_gradient(
                self.x,
                idx=range(self.loss.n)
            )
            self.norm_grad_start_epoch = self.loss.norm(grad_start_epoch)
        self.g = np.zeros_like(self.x)

    def calculate_outer_step_size(self):
        if self.use_g_in_outer_step:
            outer_step_size = \
                1 / (self.c_1 + self.c_2 * self.loss.norm(self.g))
        else:
            outer_step_size = \
                1 / (self.c_1 + self.c_2 * self.norm_grad_start_epoch)
        return outer_step_size

    def check_convergence(self):
        if self.f_tolerance is not None:
            assert self.loss.f_opt is not None, 'loss does not have f_opt!'
            f_tolerance_met = \
                self.loss.value(self.x) - self.loss.f_opt < self.f_tolerance
        else:
            return super().check_convergence()
        return super().check_convergence() or f_tolerance_met


class ClipERR2(ClipERR):

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
        self.clip_level = c_0 / c_1
        ClipERR.__init__(
            self,
            c_0,
            c_1,
            inner_step_size,
            steps_per_permutation=steps_per_permutation,
            batch_size=batch_size,
            f_tolerance=f_tolerance,
            *args,
            **kwargs
        )

        self.lr = 1 / (2 * c_0)
        self.g = None

    def step(self):
        if self.g is None:
            self.g = np.zeros_like(self.x)
            if not self.use_g_in_outer_step:
                grad_start_epoch = self.loss.stochastic_gradient(
                    self.x,
                    idx=range(self.loss.n)
                )
                self.norm_grad_start_epoch = self.loss.norm(grad_start_epoch)

        idx, normalization = self.permute()
        self.i += self.batch_size
        self.grad = self.loss.stochastic_gradient(
            self.x, 
            idx=idx, 
            normalization=normalization
        )

        if self.i < self.loss.n:
            self.perform_inner_step()
        else:
            self.perform_outer_step()
        
        self.i %= self.loss.n
    
    def calculate_outer_step_size(self):
        if self.use_g_in_outer_step:
            outer_step_size = \
                self.lr * min(1, self.clip_level / self.loss.norm(self.g))
        else:
            outer_step_size = \
                self.lr * min(1, self.clip_level / self.norm_grad_start_epoch)
        return outer_step_size
