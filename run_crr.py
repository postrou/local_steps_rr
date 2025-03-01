import os
import argparse
from multiprocessing import Pool
from itertools import product
from functools import partial

import numpy as np
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from src.optimizers import Ig, Nesterov, ClippedIg, Sgd, Shuffling, \
    ClippedShuffling, ClERR, NASTYA
from src.loss_functions import load_quadratic_dataset, load_logreg_dataset, \
    load_fourth_order_dataset
from src.utils import get_trace, relative_round


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
    step_size,
):
    so_trace = get_trace(os.path.join(f'{trace_path}', f'so_lr_{step_size}_{n_epochs}'), loss)
    if not so_trace:
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
        so_trace.save(f'so_lr_{step_size}_{n_epochs}', trace_path)


def sgd(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
):
    sgd_trace = get_trace(os.path.join(f'{trace_path}', f'sgd_lr_{step_size}_{n_epochs}'), loss)
    if not sgd_trace:
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
        sgd_trace.save(f'sgd_lr_{step_size}_{n_epochs}', trace_path)


def ig(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
):
    ig_trace = get_trace(os.path.join(f'{trace_path}', f'ig_lr_{step_size}_{n_epochs}'), loss)
    if not ig_trace:
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
        ig_trace.save(f'ig_lr_{step_size}_{n_epochs}', trace_path)


def rr(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
):
    rr_trace = get_trace(os.path.join(f'{trace_path}', f'rr_lr_{step_size}_{n_epochs}'), loss)
    if not rr_trace:
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
        rr_trace.save(f'rr_lr_{step_size}_{n_epochs}', trace_path)


def crr(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_dir, 
    batch_size, 
    step_size,
    clip_level
):
    trace_name = f'c_{clip_level}_lr_{step_size}_rr_{n_epochs}'
    crr_trace = get_trace(os.path.join(trace_dir, trace_name), loss)
    if not crr_trace:
        crr = ClippedShuffling(
            loss=loss, 
            lr0=step_size,
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

        crr_trace.save(trace_name, trace_dir)
    else:
        print(f'CRR trace with cl={clip_level}, lr={step_size} exists!')


def crr_opt(
    loss,
    x0, 
    x_opt,
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
    clip_level,
):
    crr_opt_trace = get_trace(os.path.join(trace_path, f'c_{clip_level}_rr_opt_{n_epochs}'), loss)
    if not crr_opt_trace:
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
        crr_opt_trace.save(f'c_{clip_level}_lr_{step_size}_rr_opt_{n_epochs}', trace_path)


def cig(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
    clip_level
):
    cig_trace = get_trace(f'{trace_path}c_{clip_level}_lr_{step_size}_ig_{n_epochs}', loss)
    if not cig_trace:
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
        cig_trace.save(f'c_{clip_level}_lr_{step_size}_ig_{n_epochs}', trace_path)


def cig_opt(
    loss,
    x0, 
    x_opt,
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
    clip_level
):
    cig_opt_trace = get_trace(f'{trace_path}c_{clip_level}_lr_{step_size}_ig_opt_{n_epochs}', loss)
    if not cig_opt_trace:
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
        cig_opt_trace.save(f'c_{clip_level}_lr_{step_size}_ig_opt_{n_epochs}', trace_path)


def nastya(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    step_size,
    inner_step_size,
):
    trace_name = f'nastya_lr_{step_size}_in_lr_{inner_step_size}_{n_epochs}'
    nastya_trace = get_trace(
        os.path.join(trace_path, trace_name), 
        loss
    )
    if not nastya_trace:
        nastya = NASTYA(
            inner_step_size=inner_step_size,
            lr0=step_size,
            loss=loss, 
            it_max=stoch_it, 
            batch_size=batch_size, 
            trace_len=trace_len,
            n_seeds=n_seeds, 
        )
        try:
            nastya_trace = nastya.run(x0=x0)
        except AssertionError:
            print(f'NASTYA, some error, skipping lr={step_size}, inner_lr={inner_step_size}')
            return
        nastya_trace.convert_its_to_epochs(batch_size=batch_size)
        nastya_trace.compute_loss_of_iterates()
        nastya_trace.save(
            trace_name,
            trace_path
        )
        print(f'🥰🥰🥰 Finished NASTYA trace with lr={step_size}, inner_lr={inner_step_size}! 🥰🥰🥰')
    else:
        print(f'🤙🤙🤙 NASTYA trace with lr={step_size}, inner_lr={inner_step_size} already exists! 🤙🤙🤙')


def clerr(
    loss,
    x0, 
    n_epochs, 
    stoch_it, 
    n_seeds, 
    trace_len, 
    trace_path, 
    batch_size, 
    use_g,
    inner_step_size,
    clip_level=None,
    step_size=None,
    c_0=None,
    c_1=None
):
    if use_g:
        if c_0 is None and c_1 is None:
            trace_name = f'clerr_g_c_{clip_level}_lr_{step_size}_in_lr_{inner_step_size}_{n_epochs}'
        else:
            trace_name = f'clerr_g_c_0_{c_0}_c_1_{c_1}_in_lr_{inner_step_size}_{n_epochs}'
    else:
        if c_0 is None and c_1 is None:
            trace_name = f'clerr_c_{clip_level}_lr_{step_size}_in_lr_{inner_step_size}_{n_epochs}'
        else:
            trace_name = f'clerr_c_0_{c_0}_c_1_{c_1}_in_lr_{inner_step_size}_{n_epochs}'

    clerr_trace = get_trace(
        os.path.join(trace_path, trace_name), 
        loss
    )
    if not clerr_trace:
        if c_0 is None and c_1 is None:
            c_0 = 1 / (2 * step_size)
            c_1 = c_0 / clip_level
        clerr = ClERR(
            c_0=c_0,
            c_1=c_1,
            inner_step_size=inner_step_size,
            loss=loss, 
            it_max=stoch_it, 
            batch_size=batch_size, 
            trace_len=trace_len,
            n_seeds=n_seeds, 
            f_tolerance=None,
            use_g_in_outer_step=use_g
        )
        try:
            clerr_trace = clerr.run(x0=x0)
        except AssertionError:
            if c_0 is None and c_1 is None:
                print(f'CLERR, some error, skipping cl={clip_level}, lr={step_size}, inner_lr={inner_step_size}')
            else:
                print(f'CLERR, some error, skipping c_0={c_0}, c_1={c_1}, inner_lr={inner_step_size}')
            return
        clerr_trace.convert_its_to_epochs(batch_size=batch_size)
        clerr_trace.compute_loss_of_iterates()
        clerr_trace.save(
            trace_name,
            trace_path
        )
        if c_0 is None and c_1 is None:
            print(f'🥰🥰🥰 Finished CLERR trace with cl={clip_level}, lr={step_size}, inner_lr={inner_step_size}! 🥰🥰🥰')
        else:
            print(f'🥰🥰🥰 Finished CLERR trace with c_0={c_0}, c_1={c_1}, inner_lr={inner_step_size}! 🥰🥰🥰')
    else:
        if c_0 is None and c_1 is None:
            print(f'🤙🤙🤙 CLERR trace with cl={clip_level}, lr={step_size}, inner_lr={inner_step_size} already exists! 🤙🤙🤙')
        else:
            print(f'🤙🤙🤙 CLERR trace with c_0={c_0}, lr={c_1}, inner_lr={inner_step_size} already exists! 🤙🤙🤙')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('alg', type=str)
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--x_opt', action='store_true')
    parser.add_argument('--cl_min', type=int, default=None, help='min clip level in log scale')
    parser.add_argument('--cl_max', type=int, default=None, help='max clip level in log scale')
    parser.add_argument('--lr_min', type=int, default=None, help='min step size in log scale')
    parser.add_argument('--lr_max', type=int, default=None, help='max step size in log scale')
    parser.add_argument('--in_lr_min', type=int, default=None, help='min inner step size in log scale (for CLERR and Nastya)')
    parser.add_argument('--in_lr_max', type=int, default=None, help='max inner step size in log scale (for CLERR and Nastya)')
    parser.add_argument('--c_0_min', type=int, default=None, help='min c_0 in log scale')
    parser.add_argument('--c_0_max', type=int, default=None, help='max c_0 in log scale')
    parser.add_argument('--c_1_min', type=int, default=None, help='min c_1 in log scale')
    parser.add_argument('--c_1_max', type=int, default=None, help='max c_1 in log scale')
    parser.add_argument('--a_min', type=int, default=None, help='min alpha in log scale (for shifts)')
    parser.add_argument('--a_max', type=int, default=None, help='max alpha in log scale (for shifts)')
    parser.add_argument('--use_g', action='store_true', help='use gradient estimation g instead of the full gradient in global step size calculation of CLERR')
    parser.add_argument('--n_cpus', type=int, default=10, help='number of processes to run in parallel')
    args = parser.parse_args()

    # Get data and set all parameters
    print('Loading data')
    alg = args.alg
    dataset = args.dataset
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    if dataset.startswith('quadratic'):
        n_seeds = 10
        is_noised = dataset == 'quadratic_noised'
        loss = load_quadratic_dataset(is_noised)
        f_opt, x_opt = loss.f_opt, loss.x_opt
        n, dim = len(loss.x), 1
        L = loss.smoothness()
        if args.x_opt:
            x0 = np.array([x_opt])
        else:
            x0 = np.array([3 * 1e4])        
        n_epochs = args.n_epochs
        batch_size = args.batch_size
        n_seeds = 10
        stoch_it = n_epochs * n // batch_size
        trace_len = 500

        if x0 == x_opt:
            trace_path = f'results/{dataset}/x0_x_opt/bs_{batch_size}/'
            plot_path = f'plots/{dataset}/x0_x_opt/bs_{batch_size}'
        else:
            trace_path = f'results/{dataset}/x0_{x0[0]}/bs_{batch_size}/'
            plot_path = f'plots/{dataset}/x0_{x0[0]}/bs_{batch_size}'
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)

    elif dataset.startswith('logreg'):
        dataset_name = dataset.split('_')[1]
        loss = load_logreg_dataset(dataset_name)
        n, dim = loss.A.shape
        l2 = loss.l2
        n_epochs = args.n_epochs
        batch_size = args.batch_size
        # n_seeds = 2 # was set to 20 in the paper
        n_seeds = 10
        stoch_it = n_epochs * n // batch_size
        trace_len = 500
        if dataset == 'logreg_w8a':
            # clip_level_list = np.logspace(-3, 2, 6)
            x0 = csc_matrix((dim, 1))
            trace_path = f'results/log_reg_{dataset}_l2_{relative_round(l2)}/'
        elif dataset == 'logreg_covtype':
            # clip_level_list = np.logspace(-1, 2, 4)
            x0 = csc_matrix(np.random.normal(0, 1, size=(dim, 1)))
            trace_path = f'results/log_reg_{dataset}_l2_{relative_round(l2)}_x0_random/'
        elif dataset == 'logreg_gisette':
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

    elif dataset == 'fourth_order':
        n_seeds = 10
        loss, x0, x_opt, n_seeds, stoch_it, trace_len, trace_path, plot_path = \
            load_fourth_order_dataset(n_epochs, n_seeds, args.batch_size)

    print('trace path:', trace_path) 

    if alg == 'rr':
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'
            
        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)

        print('Random Reshuffling')
        pool = Pool(min(len(step_size_list), args.n_cpus))
        partial_rr = partial(
            rr,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size
        )
        pool.map(partial_rr, step_size_list)

    elif alg == 'crr':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {alg}'
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'

        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)

        print('Clipped Random Reshuffling')
        print('step sizes:', step_size_list)
        print('clip levels:', clip_level_list)
        pool = Pool(min(len(clip_level_list), args.n_cpus))
        args_product = product(step_size_list, clip_level_list)
        partial_crr = partial(
            crr,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size
        )
        pool.starmap(partial_crr, args_product)

    elif alg == 'crr_opt':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {alg}'
        assert args.lr_min is None and args.lr_max is None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'
            
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)

        pool = Pool(min(len(clip_level_list), args.n_cpus))
        partial_crr_opt = partial(
            crr_opt,
            loss,
            x0, 
            x_opt,
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )
        pool.map(partial_crr_opt, clip_level_list)

    elif alg == 'so':
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'

        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)

        print('Single Reshuffling')
        pool = Pool(min(len(step_size_list), args.n_cpus))
        partial_so = partial(
            so,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size
        )
        pool.map(partial_so, step_size_list)

    elif alg == 'sgd':
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'

        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
        
        print('Regular SGD')
        pool = Pool(min(len(step_size_list), args.n_cpus))
        partial_sgd = partial(
            sgd,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size
        )
        pool.map(partial_sgd, step_size_list)

    elif alg == 'ig':
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'

        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
 
        print('Deterministic Reshuffling')
        pool = Pool(min(len(step_size_list), args.n_cpus))
        partial_ig = partial(
            ig,
            loss,
            x0, 
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size
        )

    elif alg == 'cig':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {alg}'
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'

        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)

        print('Clipped Determenistic Reshuffling')
        pool = Pool(min(len(clip_level_list), args.n_cpus))
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

    elif alg == 'cig_opt':
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {alg}'
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'

        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
        clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)

        print('Clipped Determenistic Reshuffling')
        pool = Pool(min(len(clip_level_list), args.n_cpus))
        partial_cig_opt = partial(
            cig_opt,
            loss,
            x0, 
            x_opt,
            n_epochs, 
            stoch_it, 
            n_seeds, 
            trace_len, 
            trace_path, 
            batch_size, 
            step_size_list,
        )
        pool.map(partial_cig_opt, clip_level_list)
        
    elif alg == 'nastya':
        assert args.in_lr_min is not None and args.in_lr_max is not None, \
            f'You did not provide --in_lr_min or --in_lr_max for algorithm {alg}'
        assert args.lr_min is not None and args.lr_max is not None, \
            f'You did not provide --lr_min or --lr_max for algorithm {alg}'
        
        step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
        in_step_size_list = np.logspace(args.in_lr_min, args.in_lr_max, args.in_lr_max - args.in_lr_min + 1)

        args_product = list(product(step_size_list, in_step_size_list))
        pool = Pool(min(len(args_product), args.n_cpus))
        print('step sizes:', step_size_list)
        print('inner step sizes:', in_step_size_list)

        partial_nastya = partial(
            nastya,
            loss,
            x0,
            n_epochs,
            stoch_it,
            n_seeds,
            trace_len,
            trace_path,
            batch_size,
        )
        if args.n_cpus == 1:
            for lr, in_lr in args_product:
                partial_nastya(lr, in_lr)
        else:
            pool.starmap(partial_nastya, args_product)

    elif alg == 'clerr':
        assert args.in_lr_min is not None and args.in_lr_max is not None, \
            f'You did not provide --in_lr_min or --in_lr_max for algorithm {alg}'

        print('CLERR with g' if args.use_g else 'CLERR')

        c_0_min, c_0_max = args.c_0_min, args.c_0_max
        c_1_min, c_1_max = args.c_1_min, args.c_1_max
        in_step_size_list = np.logspace(args.in_lr_min, args.in_lr_max, args.in_lr_max - args.in_lr_min + 1)
        if c_0_min is None and c_0_min is None and c_1_min is None and c_1_max is None:
            step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
            clip_level_list = np.logspace(args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1)
            args_product = list(product(in_step_size_list, clip_level_list, step_size_list))

            pool = Pool(min(len(args_product), args.n_cpus))
            print('step sizes:', step_size_list)
            print('clip levels:', clip_level_list)
            print('inner step sizes:', in_step_size_list)

            partial_clerr = partial(
                clerr,
                loss,
                x0,
                n_epochs,
                stoch_it,
                n_seeds,
                trace_len,
                trace_path,
                batch_size,
                args.use_g,
            )
            if args.n_cpus == 1:
                for in_lr, cl, lr in args_product:
                    partial_clerr(in_lr, cl, lr)
            else:
                pool.starmap(partial_clerr, args_product)
        else:
            c_0_list = np.logspace(c_0_min, c_0_max, c_0_max - c_0_min + 1)
            c_1_list = np.logspace(c_1_min, c_1_max, c_1_max - c_1_min + 1)
            step_size_list = [None]
            clip_level_list = [None]
            args_product = list(product(in_step_size_list, step_size_list, clip_level_list, c_0_list, c_1_list))

            pool = Pool(min(len(args_product), args.n_cpus))
            print('c_0:', c_0_list)
            print('c_1:', c_1_list)
            print('inner step sizes:', in_step_size_list)

            partial_clerr = partial(
                clerr,
                loss,
                x0,
                n_epochs,
                stoch_it,
                n_seeds,
                trace_len,
                trace_path,
                batch_size,
                args.use_g,
            )
            if args.n_cpus == 1:
                for in_lr, cl, lr, c_0, c_1 in args_product:
                    partial_clerr(in_lr, cl, lr, c_0, c_1)
            else:
                pool.starmap(partial_clerr, args_product)

    else:
        raise NotImplementedError(f'Unknown algorithm: {alg}')
