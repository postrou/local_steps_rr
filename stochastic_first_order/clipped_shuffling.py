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

        assert clip_level is not None, \
            'If you do not use clipping, use Shuffling class instead'
        self.clip_level = clip_level
        self.x_opt = x_opt
        self.use_new_opt = use_new_opt
        if x_opt is not None:
            self.x_opt_i = x_opt

        self.alpha_shift = alpha_shift
        if alpha_shift > 0:
            assert steps_per_permutation is np.inf, \
                'If we want to use shifts, we need permutation to be fixed'
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
                normalization = self.loss.n / self.steps_per_permutation #works only for RR
            shift_grad_opt_diff = 0
            for i in range(len(self.shifts)):
                idx = self.permutation[i]
                grad_opt = self.loss.stochastic_gradient(
                    self.x_opt, 
                    idx=idx, 
                    normalization=normalization
                )
                shift_i = self.shifts[i]
                shift_grad_opt_diff += self.loss.norm(shift_i - grad_opt) ** 2
            shift_grad_opt_diff /= len(self.shifts)
            self.trace.shift_grad_opt_diffs.append(shift_grad_opt_diff)
        
    def step(self):
        if self.it % self.steps_per_permutation == 0:
            # for shuffle once it enters here only on the 0-th iteration
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        if self.steps_per_permutation is np.inf:
            idx_perm = np.arange(self.i, self.i + self.batch_size)
            idx_perm %= self.loss.n
            normalization = self.batch_size
        else:
            idx_perm = np.arange(self.i, min(self.loss.n, self.i+self.batch_size))
            normalization = self.loss.n / self.steps_per_permutation #works only for RR
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        self.i %= self.loss.n
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)

        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
       
        if self.alpha_shift > 0:
            id_shift = self.it % len(self.shifts)
            shift = self.shifts[id_shift]
            if not np.isscalar(self.grad):
                shift = shift.reshape(-1, 1)
            if scipy.sparse.issparse(self.grad):
                shift = scipy.sparse.csr_matrix(shift)

            hat_delta = self.clip(self.grad - shift)
            self.grad_estimator = shift + hat_delta
            shift_next = np.transpose(shift + self.alpha_shift * hat_delta) 
            if scipy.sparse.issparse(shift_next):
                shift_next = shift_next.toarray()
            id_shift_next = (id_shift + 1) % len(self.shifts)
            self.shifts[id_shift_next] = shift_next
        elif self.x_opt is None:
            self.grad_estimator = self.clip(self.grad)
        else:
            grad_opt = self.loss.stochastic_gradient(self.x_opt, idx=idx, normalization=normalization)
            if self.use_new_opt:
                self.x_opt_i -= self.lr * self.clip(grad_opt)
                grad_opt_i = self.loss.stochastic_gradient(self.x_opt_i, idx=idx, normalization=normalization)
                self.grad_estimator = grad_opt_i + self.clip(self.grad - grad_opt_i)
            else:
                self.grad_estimator = grad_opt + self.clip(self.grad - grad_opt)

        self.x -= self.lr * self.grad_estimator
        
    def clip(self, grad):
        return min(1, self.clip_level / self.loss.norm(grad)) * grad


class ClippedShuffling2(ClippedShuffling):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
            )
        self.prev_grad = None
       
    def step(self):
        if self.it % self.steps_per_permutation == 0:
            # for shuffle once it enters here only on the 0-th iteration
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        if self.steps_per_permutation is np.inf:
            idx_perm = np.arange(self.i, self.i + self.batch_size)
            idx_perm %= self.loss.n
            normalization = self.batch_size
        else:
            idx_perm = np.arange(self.i, min(self.loss.n, self.i+self.batch_size))
            normalization = self.loss.n / self.steps_per_permutation #works only for RR
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        self.i %= self.loss.n
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)

        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)

        if self.prev_grad is None:
            self.grad_estimator = self.clip(self.grad)
        else:
            self.grad_estimator = self.prev_grad + self.clip(self.grad - self.prev_grad)
        self.prev_grad = self.grad.copy()

        self.x -= self.lr * self.grad_estimator


class ClippedShuffling3(ClippedShuffling):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
            )
        self.prev_grad = None
       
    def step(self):
        if self.it % self.steps_per_permutation == 0:
            # for shuffle once it enters here only on the 0-th iteration
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        if self.steps_per_permutation is np.inf:
            idx_perm = np.arange(self.i, self.i + self.batch_size)
            idx_perm %= self.loss.n
            normalization = self.batch_size
        else:
            idx_perm = np.arange(self.i, min(self.loss.n, self.i+self.batch_size))
            normalization = self.loss.n / self.steps_per_permutation #works only for RR
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        self.i %= self.loss.n
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)

        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)

        if self.grad_estimator is None and self.prev_grad is None:
            self.grad_estimator = self.clip(self.grad)
        else:
            self.grad_estimator = self.grad_estimator + self.clip(self.grad - self.prev_grad)
        self.prev_grad = self.grad.copy()

        self.x -= self.lr * self.grad_estimator
