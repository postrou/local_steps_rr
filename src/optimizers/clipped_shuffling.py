import math
import numpy as np
import scipy

from .shuffling import Shuffling


class ClippedShuffling(Shuffling):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """

    def __init__(
        self,
        steps_per_permutation=None,
        lr0=None,
        lr_max=np.inf,
        lr_decay_coef=0,
        lr_decay_power=1,
        it_start_decay=None,
        batch_size=1,
        clip_level=None,
        x_opt=None,
        use_new_opt=False,
        alpha_shift=0,
        *args,
        **kwargs
    ):
        super().__init__(
            steps_per_permutation,
            lr0,
            lr_max,
            lr_decay_coef,
            lr_decay_power,
            it_start_decay,
            batch_size,
            *args,
            **kwargs
        )

        assert (
            clip_level is not None
        ), "If you do not use clipping, use Shuffling class instead"
        self.clip_level = clip_level
        self.x_opt = x_opt
        self.use_new_opt = use_new_opt
        if x_opt is not None:
            self.x_opt_i = x_opt

        self.alpha_shift = alpha_shift
        if alpha_shift > 0:
            assert (
                steps_per_permutation is np.inf
            ), "If we want to use shifts, we need permutation to be fixed"
            n_shifts = math.ceil(self.loss.n / batch_size)
            self.shifts = np.zeros((n_shifts, self.loss.dim))
        else:
            self.shifts = None

    def update_trace(self):
        super().update_trace()
        if self.shifts is not None:
            if self.steps_per_permutation is np.inf:
                normalization = self.batch_size
            else:
                normalization = (
                    self.loss.n / self.steps_per_permutation
                )  # works only for RR
            shift_grad_opt_diff = 0
            for i in range(len(self.shifts)):
                idx = self.permutation[i]
                grad_opt = self.loss.stochastic_gradient(
                    self.x_opt, idx=idx, normalization=normalization
                )
                shift = self.shifts[i]
                shift_grad_opt_diff += self.loss.norm(shift - grad_opt) ** 2
            shift_grad_opt_diff /= len(self.shifts)
            self.trace.shift_grad_opt_diffs.append(shift_grad_opt_diff)

    def step(self):
        idx, normalization = self.permute()
        self.i += self.batch_size
        self.i %= self.loss.n
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        self.grad = self.loss.stochastic_gradient(
            self.x, idx=idx, normalization=normalization
        )

        self.lr = self.compute_lr()

        if self.alpha_shift > 0:
            self.perform_shift()
        elif self.x_opt is None:
            self.grad_estimator = self.grad
            self.clip_coefficient = self.calculate_clip_coefficient(self.grad)
            self.x -= self.lr * self.clip_coefficient * self.grad_estimator
        else:
            self.grad_estimator = self.grad
            grad_opt = self.loss.stochastic_gradient(
                self.x_opt, idx=idx, normalization=normalization
            )
            if self.use_new_opt:
                self.x_opt_i -= self.lr * self.clip(grad_opt)
                grad_opt_i = self.loss.stochastic_gradient(
                    self.x_opt_i, idx=idx, normalization=normalization
                )
                self.clip_coefficient = self.calculate_clip_coefficient(self.grad - grad_opt_i)
                self.x -= self.lr * (grad_opt_i + self.clip_coefficient * (self.grad_estimator - grad_opt_i))
            else:
                self.clip_coefficient = self.calculate_clip_coefficient(self.grad - grad_opt)
                self.x -= self.lr * (grad_opt + self.clip_coefficient * (self.grad_estimator - grad_opt))

    def perform_shift(self):
        id_shift = self.it % len(self.shifts)
        shift = self.shifts[id_shift]
        if not np.isscalar(self.grad):
            shift = shift.reshape(-1, 1)
        if scipy.sparse.issparse(self.grad):
            shift = scipy.sparse.csr_matrix(shift)

        hat_delta = self.clip(self.grad - shift)
        self.grad_estimator = shift + hat_delta
        shift_next_epoch = np.transpose(shift + self.alpha_shift * hat_delta)
        if scipy.sparse.issparse(shift_next_epoch):
            shift_next_epoch = shift_next_epoch.toarray()
        self.shifts[id_shift] = shift_next_epoch

    def clip(self, grad):
        self.clip_coefficient = self.calculate_clip_coefficient(grad)
        return self.clip_coefficient * grad

    def calculate_clip_coefficient(self, grad):
        return min(1, self.clip_level / self.loss.norm(grad))
