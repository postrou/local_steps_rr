import numpy as np

from .loss_oracle import Oracle


class FourthOrder(Oracle):
    def __init__(self, x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        assert len(x.shape) == 1
        self.n = len(x)
        self.dim = 1
        self.x_last = 0.

        self.x_opt = None
        self.f_opt = None

    def value(self, x):
        y = np.mean([(x - x_0) ** 4 for x_0 in self.x])
        return y

    def partial_value(self, x, idx):
        y = np.mean([(x - x_0) ** 4 for x_0 in self.x[idx]])
        return y

    def gradient(self, x):
        grad = np.mean([4 * (x - x_0) ** 3 for x_0 in self.x])
        return grad

    def stochastic_gradient(
        self, 
        x, 
        idx=None, 
        batch_size=1, 
        replace=False, 
        normalization=None
    ):
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=False)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if np.isscalar(idx):
            x_0_list = [self.x[idx]]
        else:
            x_0_list = self.x[idx]
        stoch_grad = np.mean([4 * (x - x_0) ** 3 for x_0 in x_0_list])
        return stoch_grad
    
    def norm(self, x):
        return abs(x)
    
    def smoothness(self):
        raise Exception('This function is not smooth!')
