import numpy as np

from src.optimizers.optimizer import Optimizer


class ClippedIg(Optimizer):
    """
    Incremental gradient descent (IG) with decreasing or constant learning rate.
    
    Arguments:
        lr0 (float, optional): an estimate of the inverse maximal smoothness constant
    """
    def __init__(
        self, 
        clip_level, 
        lr0=None, 
        lr_max=np.inf, 
        lr_decay_coef=0, 
        lr_decay_power=1, 
        it_start_decay=None, 
        batch_size=1, 
        x_opt=None, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_level = clip_level 
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        self.x_opt = x_opt
        
    def step(self):
        idx = np.arange(self.i, self.i + self.batch_size)
        idx %= self.loss.n
        self.i += self.batch_size
        self.i %= self.loss.n
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx)

        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)

        if self.x_opt is None:
            self.grad_estimator = self.clip(self.grad)
        else:
            grad_opt = self.loss.stochastic_gradient(self.x_opt, idx=idx)
            self.grad_estimator = grad_opt + self.clip(self.grad - grad_opt)
    
        self.x -= self.lr * self.grad_estimator

    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(self.batch_size)
        self.i = 0

    def clip(self, grad):
        return min(1, self.clip_level / self.loss.norm(grad)) * grad

