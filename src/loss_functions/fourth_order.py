import os

import numpy as np

from .loss_oracle import Oracle


class FourthOrder(Oracle):
    def __init__(self, x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_0 = x
        assert len(x.shape) == 1
        self.n = len(x)
        self.dim = 1
        self.x_last = 0.

        self.x_opt = None
        self.f_opt = None

    def value(self, x):
        y = np.mean([(x - x_0) ** 4 for x_0 in self.x_0])
        return y

    def partial_value(self, x, idx):
        y = np.mean([(x - x_0) ** 4 for x_0 in self.x_0[idx]])
        return y

    def gradient(self, x):
        grad = np.mean([4 * (x - x_0) ** 3 for x_0 in self.x_0])
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
            x_0_list = [self.x_0[idx]]
        else:
            x_0_list = self.x_0[idx]
        stoch_grad = np.mean([4 * (x - x_0) ** 3 for x_0 in x_0_list])
        return stoch_grad
    
    def norm(self, x):
        if type(x) is list or type(x) is np.ndarray:
            return abs(x[0])
        return abs(x)
    
    def smoothness(self):
        raise Exception('This function is not smooth!')

        
def find_optimal_point(loss, x0):
    step_size = 1

    x = x0.copy()
    N = 20
    for i in range(N):
        grad = loss.gradient(x)
        hess = np.mean(12 * (loss.x_0 - x)**2)
        x -= step_size * 1 / hess * grad
        f_value = loss.value(x)
    
    x_opt = x[0]
    f_opt = f_value
    return x_opt, f_opt


def load_fourth_order_dataset(n_epochs, n_seeds=None, batch_size=None, trace_len=None, save_results=True):
    np.random.seed(0)
    x = np.random.uniform(-10, 10, 1000)
    n = len(x)
    x0 = np.array([1000.0])
    loss = FourthOrder(x) 
    x_opt, f_opt = find_optimal_point(loss, x0)
    loss.x_opt, loss.f_opt = x_opt, f_opt

    n_seeds = n_seeds if n_seeds is not None else 10
    if batch_size is None:
        batch_size = n
    stoch_it = np.ceil(n / batch_size) * n_epochs
    if trace_len is None:
        trace_len = stoch_it * 0.25

    x_opt = loss.x_opt
    if save_results:
        if x0 == x_opt:
            trace_path = f'results/fourth_order/x0_x_opt/bs_{batch_size}/'
            plot_path = f'plots/fourth_order/x0_x_opt/bs_{batch_size}'
        else:
            trace_path = f'results/fourth_order/x0_{x0[0]}/bs_{batch_size}/'
            plot_path = f'plots/fourth_order/x0_{x0[0]}/bs_{batch_size}'
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
    else:
        trace_path, plot_path = None, None

    return loss, x0, x_opt, n_seeds, stoch_it, trace_len, trace_path, plot_path
