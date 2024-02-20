import os
import argparse
from multiprocessing import Pool
from itertools import product
from functools import partial
from datetime import datetime

import numpy as np
import numpy.linalg as la
from scipy.sparse import csc_matrix, csr_matrix
from tqdm.auto import tqdm

from datasets import get_dataset
from first_order import Ig, Nesterov, ClippedIg
from loss_functions import LogisticRegression, Quadratic
from stochastic_first_order import Sgd, Shuffling, ClippedShuffling
from utils import get_trace, relative_round


def best_trace_by_step_size(traces, step_size_list):
    min_i, min_val = 0, np.inf
    for i, tr in enumerate(traces):
        if tr.loss_vals is None:
            mean_loss_val = 0
            for loss_vals in tr.loss_vals_all.values():
                mean_loss_val += np.mean(loss_vals)
            mean_loss_val /= len(tr.loss_vals_all)
        else:
            mean_loss_val = np.mean(tr.loss_vals)
        if mean_loss_val < min_val:
            min_val = mean_loss_val
            min_i = i
    best_trace = traces[min_i]
    best_trace.step_size = step_size_list[min_i]
    return best_trace


def so(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
):
    so_trace = get_trace(os.path.join(f'{trace_path}', f'so_{n_epochs}'), loss)
    if not so_trace:
        so_traces = []
        for step_size in tqdm(step_size_list):
            lr0 = step_size
            so = Shuffling(
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                n_seeds=n_seeds, 
                batch_size=batch_size, 
                steps_per_permutation=np.inf, 
                trace_len=trace_len
            )
            so_trace = so.run(x0=x0)
            so_trace.convert_its_to_epochs(batch_size=batch_size)
            so_trace.compute_loss_of_iterates()
            so_traces.append(so_trace)
        so_trace = best_trace_by_step_size(so_traces, step_size_list)
    print(f'best step size: {so_trace.step_size}')
    so_trace.save(f'so_{n_epochs}', trace_path)


def sgd(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
):
    sgd_trace = get_trace(os.path.join(f'{trace_path}', f'sgd_{n_epochs}'), loss)
    if not sgd_trace:
        sgd_traces = []
        for step_size in tqdm(step_size_list):
            lr0 = step_size
            sgd = Sgd(
                loss=loss,
                lr0=lr0, 
                it_max=stoch_it, 
                n_seeds=n_seeds, 
                batch_size=batch_size, 
                avoid_cache_miss=True, 
                trace_len=trace_len
            )
            sgd_trace = sgd.run(x0=x0)
            sgd_trace.convert_its_to_epochs(batch_size=batch_size)
            sgd_trace.compute_loss_of_iterates()
            sgd_traces.append(sgd_trace)
        sgd_trace = best_trace_by_step_size(sgd_traces, step_size_list)
    print(f'best step size: {sgd_trace.step_size}')
    sgd_trace.save(f'sgd_{n_epochs}', trace_path)


def ig(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
):
    ig_trace = get_trace(os.path.join(f'{trace_path}', f'ig_{n_epochs}'), loss)
    if not ig_trace:
        ig_traces = []
        for step_size in tqdm(step_size_list):
            lr0 = step_size
            ig = Ig(
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                batch_size=batch_size, 
                trace_len=trace_len
            )
            ig_trace = ig.run(x0=x0)
            ig_trace.convert_its_to_epochs(batch_size=batch_size)
            ig_trace.compute_loss_of_iterates()
            ig_traces.append(ig_trace)
        ig_trace = best_trace_by_step_size(ig_traces, step_size_list)
    print(f'best step size: {ig_trace.step_size}')
    ig_trace.save(f'ig_{n_epochs}', trace_path)
 

def rr(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
):
    rr_trace = get_trace(os.path.join(f'{trace_path}', f'rr_{n_epochs}'), loss)
    if not rr_trace:
        rr_traces = []
        for step_size in tqdm(step_size_list):
            lr0 = step_size
            rr = Shuffling(
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                n_seeds=n_seeds, 
                batch_size=batch_size, 
                trace_len=trace_len
            )
            rr_trace = rr.run(x0=x0)
            rr_trace.convert_its_to_epochs(batch_size=batch_size)
            rr_trace.compute_loss_of_iterates()
            rr_traces.append(rr_trace)
        rr_trace = best_trace_by_step_size(rr_traces, step_size_list)
        print(f'Best step size: {rr_trace.step_size}')
        rr_trace.save(f'rr_{n_epochs}', trace_path)


def crr(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
    clip_level
):
    crr_trace = get_trace(os.path.join(trace_path, f'c_{clip_level}_rr_{n_epochs}'), loss)
    if not crr_trace:
        cl_crr_traces = []
        for step_size in step_size_list:
            lr0 = step_size
            crr = ClippedShuffling(
                loss=loss, 
                lr0=lr0,
                it_max=stoch_it, 
                n_seeds=n_seeds, 
                batch_size=batch_size, 
                trace_len=trace_len,
                clip_level=clip_level,
                steps_per_permutation=np.inf
            )
            crr_trace = crr.run(x0=x0)
            crr_trace.convert_its_to_epochs(batch_size=batch_size)
            crr_trace.compute_loss_of_iterates()
            crr_trace.compute_last_iterate_grad_norms()
            cl_crr_traces.append(crr_trace)

        crr_trace = best_trace_by_step_size(cl_crr_traces, step_size_list)
        print(f'Best step size for clip level {clip_level}: {crr_trace.step_size}')
        crr_trace.save(f'c_{clip_level}_rr_{n_epochs}', trace_path)


def crr_opt(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
    clip_level
):
    crr_opt_trace = get_trace(os.path.join(trace_path, f'c_{clip_level}_rr_opt_{n_epochs}'), loss)
    if not crr_opt_trace:
        cl_crr_opt_traces = []
        for step_size in step_size_list:
            lr0 = step_size
            crr_opt = ClippedShuffling(
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                n_seeds=n_seeds, 
                batch_size=batch_size, 
                trace_len=trace_len,
                clip_level=clip_level,
                x_opt=x_opt,
                steps_per_permutation=np.inf
            )
            crr_opt_trace = crr_opt.run(x0=x0)
            crr_opt_trace.convert_its_to_epochs(batch_size=batch_size)
            crr_opt_trace.compute_loss_of_iterates()
            crr_opt_trace.compute_last_iterate_grad_norms()
            cl_crr_opt_traces.append(crr_opt_trace)

        crr_opt_trace = best_trace_by_step_size(cl_crr_opt_traces, step_size_list)
        print(f'Best step size for clip level {clip_level}: {crr_opt_trace.step_size}')
        crr_opt_trace.save(f'c_{clip_level}_rr_opt_{n_epochs}', trace_path)


def crr_shift(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
    alpha_shift, 
    clip_level
):
    crr_shift_trace = get_trace(os.path.join(f'{trace_path}', f'c_{clip_level}_a_shift_{alpha_shift}_rr_{n_epochs}'), loss)
    if not crr_shift_trace:        
        cl_crr_shift_traces = []
        for step_size in step_size_list:
            lr0 = step_size
            crr_shift = ClippedShuffling(
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                n_seeds=n_seeds, 
                batch_size=batch_size, 
                trace_len=trace_len,
                clip_level=clip_level,
                alpha_shift=alpha_shift,
                steps_per_permutation=np.inf
            )
            crr_shift_trace = crr_shift.run(x0=x0)
            crr_shift_trace.convert_its_to_epochs(batch_size=batch_size)
            crr_shift_trace.compute_loss_of_iterates()
            crr_shift_trace.compute_last_iterate_grad_norms()
            cl_crr_shift_traces.append(crr_shift_trace)

        crr_shift_trace = best_trace_by_step_size(cl_crr_shift_traces, step_size_list)
        print(f'Best step size for alpha {alpha_shift}, clip level {clip_level}: {crr_shift_trace.step_size}')
        crr_shift_trace.save(f'c_{clip_level}_a_shift_{alpha_shift}_rr_{n_epochs}', trace_path)

def cig(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
    clip_level
):
    cig_trace = get_trace(f'{trace_path}c_{clip_level}_ig_{n_epochs}', loss)
    if not cig_trace:
        cl_cig_traces = []
        for step_size in step_size_list:
            lr0 = step_size
            cig = ClippedIg(
                clip_level,
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                batch_size=batch_size, 
                trace_len=trace_len
            )
            cig_trace = cig.run(x0=x0)
            cig_trace.convert_its_to_epochs(batch_size=batch_size)
            cig_trace.compute_loss_of_iterates()
            cl_cig_traces.append(cig_trace)
    
        cig_trace = best_trace_by_step_size(cl_cig_traces, step_size_list)
        print(f'best step size: {cig_trace.step_size}')
        cig_trace.save(f'c_{clip_level}_ig_{n_epochs}', trace_path)
 

def cig_opt(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size_list,
    clip_level
):
    cig_opt_trace = get_trace(f'{trace_path}c_{clip_level}_ig_opt_{n_epochs}', loss)
    if not cig_opt_trace:
        cl_cig_opt_traces = []
        for step_size in step_size_list:
            lr0 = step_size
            cig_opt = ClippedIg(
                clip_level,
                loss=loss, 
                lr0=lr0, 
                it_max=stoch_it, 
                batch_size=batch_size, 
                trace_len=trace_len,
                x_opt=x_opt
            )
            cig_opt_trace = cig_opt.run(x0=x0)
            cig_opt_trace.convert_its_to_epochs(batch_size=batch_size)
            cig_opt_trace.compute_loss_of_iterates()
            cl_cig_opt_traces.append(cig_opt_trace)

        cig_opt_trace = best_trace_by_step_size(cl_cig_opt_traces, step_size_list)
        print(f'best step size: {cig_opt_trace.step_size}')
        cig_opt_trace.save(f'c_{clip_level}_ig_opt_{n_epochs}', trace_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('alg', type=str)
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--x_opt', action='store_true')
    parser.add_argument('--cl_min', type=int, default=None, help='min clip level in log scale')
    parser.add_argument('--cl_max', type=int, default=None, help='max clip level in log scale')
    parser.add_argument('--a_min', type=int, default=None, help='min alpha in log scale')
    parser.add_argument('--a_max', type=int, default=None, help='max alpha in log scale')
    args = parser.parse_args()

    # Get data and set all parameters
    print('Loading data')
    dataset = args.dataset
    if dataset.startswith('quadratic'):
        np.random.seed(0)
        if dataset == 'quadratic':
            x = np.append(np.random.randint(0, 501, 500), 
                          np.random.randint(1e5 - 500, 1e5 + 1, 500))
        elif dataset == 'quadratic_noised':
            n_left_functions = 300
            var = 3
            x = np.append(
                np.random.randint(0, 501, n_left_functions) + np.random.normal(0, var, size=n_left_functions), 
                np.random.randint(1e5 - 500, 1e5 + 1, 1000 - n_left_functions) + \
                    np.random.normal(0, var, size=1000 - n_left_functions))
        else:
            raise NotImplementedError(f'Dataset {dataset} is not implemented!')

        loss = Quadratic(x)
        f_opt = loss.f_opt
        x_opt = loss.x_opt
        n, dim = len(x), 1
        L = loss.smoothness()
        if args.x_opt:
            x0 = np.array([x_opt])
        else:
            x0 = np.array([3 * 1e4])        
        n_epochs = args.n_epochs
        batch_size = 32
        # n_seeds = 2 # was set to 20 in the paper
        n_seeds = 10
        stoch_it = n_epochs * n // batch_size
        trace_len = 300

        if x0 == x_opt:
            trace_path = f'results/{dataset}/x0_x_opt/bs_{batch_size}/'
            plot_path = f'plots/{dataset}/x0_x_opt/bs_{batch_size}'
        else:
            trace_path = f'results/{dataset}/x0_{x0[0]}/bs_{batch_size}/'
            plot_path = f'plots/{dataset}/x0_{x0[0]}/bs_{batch_size}'
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)

        step_size_list = np.logspace(-5, 0, 6)

    else:
        A, b = get_dataset(dataset)
        loss = LogisticRegression(A, b, l1=0, l2=0)
        n, dim = A.shape
        L = loss.smoothness()
        l2 = L / np.sqrt(n)
        loss.l2 = l2
        n_epochs = args.n_epochs
        batch_size = 512
        # n_seeds = 2 # was set to 20 in the paper
        n_seeds = 10
        stoch_it = n_epochs * n // batch_size
        trace_len = 300
        if dataset == 'w8a':
            # clip_level_list = np.logspace(-3, 2, 6)
            x0 = csc_matrix((dim, 1))
            trace_path = f'results/log_reg_{dataset}_l2_{relative_round(l2)}/'
        elif dataset == 'covtype':
            # clip_level_list = np.logspace(-1, 2, 4)
            x0 = csc_matrix(np.random.normal(0, 1, size=(dim, 1)))
            trace_path = f'results/log_reg_{dataset}_l2_{relative_round(l2)}_x0_random/'
        elif dataset == 'gisette':
            # clip_level_list = np.logspace(-1, 2, 4)
            x0 = csc_matrix((dim, 1))
            trace_path = f'results/log_reg_{dataset}_l2_{relative_round(l2)}/'
        else:
            raise Exception(f'The parameters for dataset {dataset} are not set up!')

        # Run the methods
        print('Finding optimal solution with FGM')
        nest_str_trace = get_trace(f'{trace_path}nest_str', loss)
        if not nest_str_trace:
            nest_str = Nesterov(loss=loss, it_max=n_epochs, mu=l2, strongly_convex=True)
            nest_str_trace = nest_str.run(x0=x0)
            nest_str_trace.compute_loss_of_iterates()
            nest_str_trace.save('nest_str', trace_path)
        f_opt = np.min(nest_str_trace.loss_vals)
        x_opt = nest_str_trace.xs[-1]

    print(trace_path)
    print(plot_path)

    algorithm = args.alg

    if algorithm == 'rr':
        print('Random Reshuffling')
        rr(
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )

    elif algorithm == 'crr':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {algorithm}'
        print('Clipped Random Reshuffling')
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
        pool = Pool(min(len(clip_level_list), 50))
        partial_crr = partial(
            crr,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )
        pool.map(partial_crr, clip_level_list)
        
    elif algorithm == 'crr_opt':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {algorithm}'
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
        pool = Pool(min(len(clip_level_list), 50))
        partial_crr_opt = partial(
            crr_opt,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )
        pool.map(partial_crr_opt, clip_level_list)

    elif algorithm == 'crr_shift':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {algorithm}'
        assert args.a_min is not None and args.a_max is not None, \
            f'You did not provide --a_min or --a_max for algorithm {algorithm}'
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
        alpha_shift_list = np.logspace(args.a_min, args.a_max, args.a_max - args.a_min + 1)
        alpha_cl_list = list(product(alpha_shift_list, clip_level_list))
        print('Clipping with shifts')
        pool = Pool(min(len(alpha_cl_list), 50))
        partial_crr_shift = partial(
            crr_shift,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list
        )
        pool.starmap(partial_crr_shift, alpha_cl_list)

    elif algorithm == 'so':
        print('Single Reshuffling')
        so(
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list
        )

    elif algorithm == 'sgd':
        print('Regular SGD')
        sgd(
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )

    elif algorithm == 'ig':
        print('Deterministic Reshuffling')
        ig(
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )

    elif algorithm == 'cig':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {algorithm}'
        print('Clipped Determenistic Reshuffling')
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
        pool = Pool(min(len(clip_level_list), 50))
        partial_cig = partial(
            cig,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )
        pool.map(partial_cig, clip_level_list)

    elif algorithm == 'cig_opt':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {algorithm}'
        print('Clipped Determenistic Reshuffling')
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
        pool = Pool(min(len(clip_level_list), 50))
        partial_cig_opt = partial(
            cig_opt,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )
        pool.map(partial_cig_opt, clip_level_list)

