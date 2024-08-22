import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle

from src.loss_functions import safe_sparse_norm


class StochasticTrace:
    """
    Class that stores the logs of running a stochastic
    optimization method and plots the trajectory.
    """
    def __init__(self, loss, is_nn=False):
        self.loss = loss
        self.is_nn = is_nn
        if not is_nn:
            self.xs_all = {}
            self.loss_is_computed = False
        self.ts_all = {}
        self.its_all = {}
        self.loss_vals_all = {}
        self.grad_norms_last_iterate = {}
        self.its_converted_to_epochs = False
        self.step_size = None
        self.clip_level = None
        self.grad_estimators_norms_all = {}
        self.shift_grad_opt_diffs_all = {}
        
    def init_seed(self):
        if self.is_nn:
            self.xs = []
            self.loss_vals = None
        else:
            self.loss_vals = []
            self.xs = None
        self.ts = []
        self.its = []
        self.grad_estimators_norms = []
        self.shift_grad_opt_diffs = []
        
    def append_seed_results(self, seed):
        if self.is_nn:
            assert self.loss_vals is not None
            self.loss_vals_all[seed] = self.loss_vals.copy()
        else:
            self.xs_all[seed] = self.xs.copy()
            self.loss_vals_all[seed] = self.loss_vals.copy() if self.loss_vals else None
        self.ts_all[seed] = self.ts.copy()
        self.its_all[seed] = self.its.copy()
        self.grad_estimators_norms_all[seed] = self.grad_estimators_norms.copy()
        self.shift_grad_opt_diffs_all[seed] = self.shift_grad_opt_diffs.copy()
    
    def compute_loss_of_iterates(self):
        if self.is_nn:
            raise Exception('Does no work for neural networks!')
        for seed, loss_vals in self.loss_vals_all.items():
            if loss_vals is None:
                self.loss_vals_all[seed] = np.asarray([self.loss.value(x) for x in self.xs_all[seed]])
            else:
                print("""Loss values for seed {} have already been computed. 
                      Set .loss_vals_all[{}] = None to recompute""".format(seed, seed))
        self.loss_is_computed = True
    
    def compute_last_iterate_grad_norms(self):
        self.grad_norms_last_iterate = {}
        for seed in self.xs_all:
            self.grad_norms_last_iterate[seed] = np.empty(self.loss.n)
            x_last = self.xs_all[seed][-1]
            for i in range(self.loss.n):
                grad_i = self.loss.stochastic_gradient(x_last, idx=i)
                grad_norm_i = self.loss.norm(grad_i)
                self.grad_norms_last_iterate[seed][i] = grad_norm_i
    
    def best_loss_value(self):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min([np.min(loss_vals) for loss_vals in self.loss_vals_all.values()])
    
    def convert_its_to_epochs(self, batch_size=1):
        its_per_epoch = self.loss.n / batch_size
        if self.its_converted_to_epochs:
            return
        for seed, its in self.its_all.items():
            self.its_all[seed] = np.asarray(its) / its_per_epoch
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
        
    def plot_losses(self, f_opt=None, log_std=True, markevery=None, alpha=0.25, ax=None, *args, **kwargs):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = self.best_loss_value()
        it_ave = np.mean([np.asarray(its) for its in self.its_all.values()], axis=0)
        if log_std:
            y_log = [np.log(loss_vals-f_opt) for loss_vals in self.loss_vals_all.values()]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            upper, lower = np.exp(y_log_ave + y_log_std), np.exp(y_log_ave - y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = [loss_vals-f_opt for loss_vals in self.loss_vals_all.values()]
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            upper, lower = y_ave + y_std, y_ave - y_std
        if markevery is None:
            markevery = max(1, len(y_ave)//20)
            
        if ax is None:
            plot = plt.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
            if len(self.loss_vals_all.keys()) > 1:
                plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
            plt.ylabel(r'$f(x)-f^*$')
        else:
            ax.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
            if len(self.loss_vals_all.keys()) > 1:
                ax.fill_between(it_ave, upper, lower, alpha=alpha, color=ax.get_lines()[-1].get_color())
            ax.set_ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, x_opt=None, log_std=True, markevery=None, alpha=0.25, ax=None, *args, **kwargs):
        if self.is_nn:
            raise Exception('Does no work for neural networks!')
        if x_opt is None:
            if self.loss_is_computed:
                f_opt = np.inf
                for seed, loss_vals in self.loss_vals_all.items():
                    i_min = np.argmin(loss_vals)
                    if loss_vals[i_min] < f_opt:
                        f_opt = loss_vals[i_min]
                        x_opt = self.xs_all[seed][i_min]
                else:
                    x_opt = self.xs[-1]
        
        it_ave = np.mean([np.asarray(its) for its in self.its_all.values()], axis=0)
        dists = [np.asarray([safe_sparse_norm(x-x_opt)**2 for x in xs]) for xs in self.xs_all.values()]
        if log_std:
            y_log = [np.log(dist) for dist in dists]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            upper, lower = np.exp(y_log_ave + y_log_std), np.exp(y_log_ave - y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = dists
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            upper, lower = y_ave + y_std, y_ave - y_std
        if markevery is None:
            markevery = max(1, len(y_ave)//20)
            
        if ax is None:
            plot = plt.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
            if len(self.loss_vals_all.keys()) > 1:
                plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
            plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        else:
            ax.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
            if len(self.loss_vals_all.keys()) > 1:
                ax.fill_between(it_ave, upper, lower, alpha=alpha, color=ax.get_lines()[-1].get_color())
            ax.set_ylabel(r'$\Vert x-x^*\Vert^2$')
            
        
    def save(self, file_name, path='./results/'):
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        f = open(path + file_name, 'wb')
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
