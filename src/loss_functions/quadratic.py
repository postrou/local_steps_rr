import numpy as np

from .loss_oracle import Oracle


class Quadratic(Oracle):
    def __init__(self, x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        assert len(x.shape) == 1
        self.n = len(x)
        self.dim = 1
        self.x_last = 0.

        self.x_opt = np.mean(x)
        self.f_opt = self.value(self.x_opt)

    def value(self, x):
        y = np.sum([(x - x_0) ** 2 for x_0 in self.x])
        return y

    def partial_value(self, x, idx):
        y = np.sum([(x - x_0) ** 2 for x_0 in self.x[idx]])
        return y

    def gradient(self, x):
        grad = np.sum([2 * (x - x_0) for x_0 in self.x])
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
        stoch_grad = np.mean([2 * (x - x_0) for x_0 in x_0_list])
        return stoch_grad
    
    def norm(self, x):
        return abs(x)
    
    def smoothness(self):
        return 2 * self.n


def load_quadratic_dataset(is_noised):
    np.random.seed(0)
    if not is_noised:
        x = np.append(np.random.randint(0, 501, 500), 
                        np.random.randint(1e5 - 500, 1e5 + 1, 500))
    else:
        n_left_functions = 300
        var = 3
        x = np.append(
            np.random.randint(0, 501, n_left_functions) + np.random.normal(0, var, size=n_left_functions), 
            np.random.randint(1e5 - 500, 1e5 + 1, 1000 - n_left_functions) + \
                np.random.normal(0, var, size=1000 - n_left_functions))

    loss = Quadratic(x)
    return loss


