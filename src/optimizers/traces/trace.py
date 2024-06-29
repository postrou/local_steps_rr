import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle

from src.loss_functions import safe_sparse_norm


class Trace:
    """
    Class that stores the logs of running an optimization method
    and plots the trajectory.
    """
    def __init__(self, loss):
        self.loss = loss
        self.xs = []
        self.ts = [] # time
        self.its = [] # iterations (can be converted to epochs)
        self.loss_vals = None
        self.its_converted_to_epochs = False
        self.loss_is_computed = False
        self.grad_estimators = []
        self.shift_grad_opt_diffs = []
        
    def compute_loss_of_iterates(self):
        if self.loss_vals is None:
            self.loss_vals = np.asarray([self.loss.value(x) for x in self.xs])
        else:
            print('Loss values have already been computed. Set .loss_vals = None to recompute')
    
    def convert_its_to_epochs(self, batch_size=1):
        its_per_epoch = self.loss.n / batch_size
        if self.its_converted_to_epochs:
            return
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
          
    def plot_losses(self, f_opt=None, ax=None, markevery=None, *args, **kwargs):
        if self.loss_vals is None:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = np.min(self.loss_vals)
        if markevery is None:
            markevery = max(1, len(self.loss_vals)//20)
        if ax is None:
            plt.plot(self.its, self.loss_vals - f_opt, markevery=markevery, *args, **kwargs)
            plt.ylabel(r'$f(x)-f^*$')
        else:
            ax.plot(self.its, self.loss_vals - f_opt, markevery=markevery, *args, **kwargs)
            ax.set_ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, x_opt=None, ax=None, markevery=None, *args, **kwargs):
        if x_opt is None:
            if self.loss_vals is None:
                x_opt = self.xs[-1]
            else:
                i_min = np.argmin(self.loss_vals)
                x_opt = self.xs[i_min]
        if markevery is None:
            markevery = max(1, len(self.xs)//20)
        dists = [safe_sparse_norm(x-x_opt)**2 for x in self.xs]
        if ax is None:
            plt.plot(self.its, dists, markevery=markevery, *args, **kwargs)
            plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        else:
            ax.plot(self.its, dists, markevery=markevery, *args, **kwargs)
            ax.set_ylabel(r'$\Vert x-x^*\Vert^2$')
        
    def best_loss_value(self):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min(self.loss_vals)
        
    def save(self, file_name, path='./results/'):
        # To make the dumped file smaller, remove the loss
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        f = open(os.path.join(path, file_name), 'wb')
        pickle.dump(self, f)
        f.close()
        
    @classmethod
    def from_pickle(cls, path, loss):
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as f:
            trace = pickle.load(f)
            trace.loss = loss
        return trace
        
        
