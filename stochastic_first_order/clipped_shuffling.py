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

        self.clip_level = clip_level
        self.x_opt = x_opt

        self.alpha_shift = alpha_shift
        if alpha_shift > 0:
            assert steps_per_permutation is np.inf, \
                'If we want to use shifts, we need permutation to be fixed'
            n_shifts = math.ceil(self.loss.n / batch_size)
            self.shifts = np.zeros((n_shifts, self.loss.dim))
        
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
        
        if self.clip_level is not None:
            if self.x_opt is None:
                if self.alpha_shift > 0:
                    id_shift = self.it % len(self.shifts)
                    shift = self.shifts[id_shift]
                    if not np.isscalar(self.grad):
                        shift = shift.reshape(-1, 1)
                    if scipy.sparse.issparse(self.grad):
                        shift = scipy.sparse.csr_matrix(shift)

                    hat_delta = self.clip(self.grad - shift)
                    self.grad = shift + hat_delta
                    shift_next = np.transpose(shift + self.alpha_shift * hat_delta) 
                    if scipy.sparse.issparse(shift_next):
                        shift_next = shift_next.toarray()
                    self.shifts[id_shift] = shift_next
                else:
                    self.grad = self.clip(self.grad)
            else:
                grad_opt = self.loss.stochastic_gradient(self.x_opt, idx=idx, normalization=normalization)
                self.grad = grad_opt + self.clip(self.grad - grad_opt)

        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
        
    def clip(self, grad):
        return min(1, self.clip_level / self.loss.norm(grad)) * grad
