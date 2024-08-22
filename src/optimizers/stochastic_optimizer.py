import time

import scipy
import numpy as np
import numpy.linalg as la

from src.loss_functions import safe_sparse_norm
from .traces import StochasticTrace
from ..utils import set_seed
from .optimizer import Optimizer


        
class StochasticOptimizer(Optimizer):
    """
    Base class for stochastic optimization algorithms. 
    The class has the same methods as Optimizer and, in addition, uses
    multiple seeds to run the experiments.
    """
    def __init__(self, loss, n_seeds=1, seeds=None, is_nn=False, *args, **kwargs):
        super(StochasticOptimizer, self).__init__(loss=loss, *args, **kwargs)
        if is_nn:
            self.tolerance = None
        self.seeds = seeds
        if not seeds:
            np.random.seed(0)
            self.seeds = [np.random.randint(100000) for _ in range(n_seeds)]
        self.finished_seeds = []
        self.trace = StochasticTrace(loss=loss, is_nn=is_nn)
    
    def run(self, x0=None):
        if not self.trace.is_nn:
            assert x0 is not None, 'x0 can be None only for neural networks!'
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            set_seed(seed)
            self.trace.init_seed()
            if not self.initialized:
                self.init_run(x0)
                self.initialized = True
            
            while not self.check_convergence():
                if not self.trace.is_nn:
                    if self.tolerance > 0:
                        self.x_old = self.x.copy()
                self.step()
                self.save_checkpoint()
                if not self.trace.is_nn:
                    assert scipy.sparse.issparse(self.x) or np.isfinite(self.x).all()

            self.trace.append_seed_results(seed)
            self.finished_seeds.append(seed)
            self.initialized = False
        return self.trace
    
    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        no_time_left = time.time()-self.t_start >= self.t_max
        if not self.trace.is_nn:
            if self.tolerance > 0:
                tolerance_met = self.x_old is not None and safe_sparse_norm(self.x-self.x_old) < self.tolerance
            else:
                tolerance_met = False
            return no_it_left or no_time_left or tolerance_met
        return no_it_left or no_time_left
 
    def init_run(self, x0=None):
        if not self.trace.is_nn:
            assert x0 is not None, 'x0 can be None only for neural networks!'
            self.x = x0.copy()
            self.trace.xs = [x0.copy()]
            self.trace.loss_vals = None
            initial_grad_norm = self.loss.norm(self.loss.gradient(x0))
        else:
            initial_loss_val = self.loss.value()
            self.trace.loss_vals = [initial_loss_val]
            initial_grad_norm = self.loss.norm(self.loss.gradient())
        self.trace.its = [0]
        self.trace.ts = [0]
        self.trace.grad_estimators_norms = [initial_grad_norm]
        self.it = 0
        self.t = 0
        self.t_start = time.time()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0
        self.grad_estimator = None

    def update_trace(self):
        self.trace.ts.append(self.t)
        if not self.trace.is_nn:
            self.trace.xs.append(self.x.copy())
        else:
            loss_val = self.loss.value()
            self.trace.loss_vals.append(loss_val)
        self.trace.ts.append(self.t)
        self.trace.its.append(self.it)
        if self.grad_estimator is None:
            self.trace.grad_estimators_norms.append(self.loss.norm(self.grad))
        else:
            self.trace.grad_estimators_norms.append(self.loss.norm(self.grad_estimator))
                