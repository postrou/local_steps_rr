import numpy as np

from .shuffling import Shuffling


class ClipERR(Shuffling):
    """ClipERR, where we don't use explicit clipping, but compute outer step size
    """

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
        self.c_0 = c_0
        self.c_1 = c_1
        self.inner_step_size = inner_step_size
        self.f_tolerance = f_tolerance
        self.use_g_in_outer_step = use_g_in_outer_step
        self.grad_estimator = None
    
    def step(self):
        if self.grad_estimator is None:
            self.grad_estimator = np.zeros_like(self.x)
            if not self.use_g_in_outer_step:
                self.update_norm_grad_start_epoch()

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
            self.i = 0
            self.sampled_permutations += 1

        idx_perm = np.arange(self.i, min(self.loss.n, self.i + self.batch_size))
        idx = self.permutation[idx_perm]
        return idx, None


    def update_norm_grad_start_epoch(self):
        grad_start_epoch = self.loss.gradient(self.x)
        self.norm_grad_start_epoch = self.loss.norm(grad_start_epoch)

    def perform_inner_step(self):
        self.x -= self.inner_step_size * self.stoch_grad

    def perform_outer_step(self):
        outer_step_size = self.calculate_outer_step_size()

        print(self.it, outer_step_size, self.x, self.grad_estimator)
        self.x -= outer_step_size * self.grad_estimator
        if not self.use_g_in_outer_step:
            self.update_norm_grad_start_epoch()
        self.grad_estimator = np.zeros_like(self.x)

    def calculate_outer_step_size(self):
        """Calculating outer step size

        Returns:
            outer_step_size: outer step size, calculated from c_0 and c_1
        """
        if self.use_g_in_outer_step:
            outer_step_size = \
                1 / (self.c_0 + self.c_1 * self.loss.norm(self.grad_estimator))
        else:
            outer_step_size = \
                1 / (self.c_0 + self.c_1 * self.norm_grad_start_epoch)
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
    """
        ClipERR with explicit clipping.
        We get step_size and clip_level from c_0 and c_1 and use outer step size
        as clipping.
    """
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
        super().__init__(
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
        self.grad_estimator = None
   
    def perform_outer_step(self):
        outer_step_size = self.calculate_outer_step_size()

        self.x -= self.lr * outer_step_size * self.grad_estimator
        if not self.use_g_in_outer_step:
            self.update_norm_grad_start_epoch()
        self.grad_estimator = np.zeros_like(self.x)

    def calculate_outer_step_size(self):
        """Calculating outer step size.
        Here outer_step_size behaves like clipping.

        Returns:
            outer_step_size: outer step size, computed from gradient norm and 
            clip_level
        """
        if self.use_g_in_outer_step:
            outer_step_size = \
                min(1, self.clip_level / self.loss.norm(self.grad_estimator))
        else:
            outer_step_size = \
                min(1, self.clip_level / self.norm_grad_start_epoch)
        return outer_step_size
