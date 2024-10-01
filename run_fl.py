import os
import argparse
from multiprocessing import Pool
import pickle
from functools import partial
from itertools import product
from copy import deepcopy

import numpy as np


def generate_data(dim=1):
    np.random.seed(0)
    data = np.random.uniform(-10, 10, (1000, dim))
    data = np.array(sorted(data, key=np.linalg.norm))
    for i in range(len(data) - 1):
        assert np.linalg.norm((data[i])) < np.linalg.norm(np.linalg.norm(data[i + 1]))
    return data


def store_result(result, result_path):
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)


def norm(x, axis=0):
    return np.sum(x ** 2, axis=axis) ** 0.5


def value(x, data_batch):
    diff = x - data_batch
    sq_norm = np.sum(diff ** 2, axis=1)
    return np.mean(sq_norm ** 2)


def gradient(x, data_batch):
    diff = x - data_batch
    sq_norm = np.sum(diff ** 2, axis=1).reshape(-1, 1)
    grad = 4 * np.mean(sq_norm * diff, axis=0)
    return grad


def hessian(x, data_batch):
    if type(x) != np.ndarray:
        x = np.array([x])
    elif len(x.shape) == 1:
        x = x.reshape(-1, 1)
    I = np.identity(len(x))
    hess = 4 * np.mean([(x - x_0) @ (x - x_0).T + np.linalg.norm(x - x_0) ** 2 * I for x_0 in data_batch], axis=0)
    return hess


def optimal_value(data, x_0=None):
    dim = data.shape[1]
    if dim == 1:
        step_size = 1

        if x_0 is None:
            x_0 = np.array([1.] * data.shape[1])
        x = x_0
        N = 100
        for i in range(N):
            grad = gradient(x, data)
            hess = hessian(x, data)
            x -= step_size * np.linalg.inv(hess) @ grad
            f_value = value(x, data)
        x_opt = x
        f_opt = f_value
    else:
        if x_0 is None:
            x_0 = np.array([1.] * data.shape[1])
        x = x_0
        N = 100
        c_0 = 10000
        c_1 = 1e-10
        x = deepcopy(x_0)
        for i in range(N):
            grad = gradient(x, data)
            lr = calculate_clipped_lr(c_0, c_1, grad)
            x -= lr * grad
        x_opt, f_opt = x, value(x, data)
    return x_opt, f_opt

def calculate_clipped_lr(c_0, c_1, g_p):
    return 1 / (c_0 + c_1 * norm(g_p))


def clip(vector, clip_level):
    return vector * min(1, clip_level / norm(vector))


def l0l1_gd(
    x_0, 
    data, 
    n_comms,
    n_seeds,
    c_0,
    c_1,
    f_opt=None,
):
    dim = len(x_0)
    results_dir = f'results/fl_fourth_order/dim_{dim}/x_0_{x_0[0]}/gd'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_fn = f'gd_c_0_{c_0}_c_1_{c_1}_{n_comms}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for l0l1_gd with c_0={c_0}, c_1={c_1} exist!')
        print(result_path)
        return

    seed_loss_vals = {}

    for seed in range(n_seeds):
        x_p = deepcopy(x_0)
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = value(x_0, data) - f_opt if f_opt is not None else value(x_0, data)
        seed_loss_vals[seed] = loss_vals_list

        for p in range(n_comms):
            g_p = gradient(x_p, data)
            server_lr = calculate_clipped_lr(c_0, c_1, g_p)
            x_p -= server_lr * g_p
            f_val = value(x_p, data) - f_opt if f_opt is not None else value(x_p, data)
            if np.isnan(f_val):
                print(f'Divergence of l0l1_gd with c_0={c_0}, c_1={c_1}')
                return
            loss_vals_list[p + 1] = f_val

    store_result(seed_loss_vals, result_path)


def local_sgd_jump(
    x_0, 
    data, 
    batch_size, 
    n_clients, 
    n_local_steps,
    n_comms,
    n_seeds,
    client_lr, 
    c_0,
    c_1,
    samp_ret=True,
    f_opt=None
):
    dim = len(x_0)
    results_dir = f"results/fl_fourth_order/dim_{dim}/x_0_{x_0[0]}/tau_{n_local_steps}/bs_{batch_size}"
    if not samp_ret:
        results_dir += '_no_ret'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_fn = f'l_sgd_jump_cl_lr_{client_lr}_c_0_{c_0}_c_1_{c_1}_{n_comms}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for local_sgd_jump with client_lr={client_lr}, c_0={c_0}, c_1={c_1} exist!')
        return

    seed_loss_vals = {}
    data_per_cl = len(data) // n_clients
    client_data_list = [data[i * data_per_cl : (i + 1) * data_per_cl] for i in range(n_clients)]
    for cld in client_data_list:
        assert len(cld) == data_per_cl
    # print(f'Number of data points per client: {data_per_cl}')
    # print(f'Client data lens: {[len(d) for d in client_data_list]}')

    for seed in range(n_seeds):
        x_p = deepcopy(x_0)
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = value(x_0, data) - f_opt if f_opt is not None else value(x_0, data)
        seed_loss_vals[seed] = loss_vals_list

        for p in range(n_comms):
            g_p = np.zeros_like(x_p)

            for m in range(n_clients):
                # local run
                x_p_m = deepcopy(x_p)
                client_data = client_data_list[m]

                for i in range(n_local_steps):
                    batch_idx = np.random.choice(len(client_data), batch_size, replace=samp_ret)
                    input_batch = client_data[batch_idx]
                    stoch_grad = gradient(x_p_m, input_batch)
                    if np.isnan(stoch_grad).any():
                        print(f'Divergence of local_sgd_jump with client_lr={client_lr}, c_0={c_0}, c_1={c_1} on p={p}, m={m}, i={i}')
                        return
                    x_p_m -= client_lr * stoch_grad

                g_p += x_p - x_p_m

            g_p *= 1 / (client_lr * n_clients * n_local_steps)
            server_lr = calculate_clipped_lr(c_0, c_1, g_p)
            x_p -= server_lr * g_p
            f_val = value(x_p, data) - f_opt if f_opt is not None else value(x_p, data)
            if np.isnan(f_val):
                print(f'Divergence of local_sgd_jump with client_lr={client_lr}, c_0={c_0}, c_1={c_1}')
                return
            loss_vals_list[p + 1] = f_val
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')

        # print()

    store_result(seed_loss_vals, result_path)


def local_clip_sgd(
    x_0, 
    data, 
    batch_size, 
    n_clients, 
    n_local_steps,
    n_comms,
    n_seeds,
    c_0,
    c_1,
    samp_ret=True,
    f_opt=None,
):
    dim = len(x_0)
    results_dir = f'results/fl_fourth_order/dim_{dim}/x_0_{x_0[0]}/tau_{n_local_steps}/bs_{batch_size}'
    if not samp_ret:
        results_dir += '_no_ret'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_fn = f'l_clip_sgd_c_0_{c_0}_c_1_{c_1}_{n_comms}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for l_clip_sgd with c_0={c_0}, c_1={c_1} exist!')
        return

    seed_loss_vals = {}
    data_per_cl = len(data) // n_clients
    client_data_list = [data[i * data_per_cl : (i + 1) * data_per_cl] for i in range(n_clients)]
    for cld in client_data_list:
        assert len(cld) == data_per_cl
    # print(f'Number of data points per client: {data_per_par}')
    # print(f'Client data lens: {[len(d) for d in client_data_list]}')
    # print(f'Comm [{0}/{n_comms}] f-f_opt={value(x_p, data) - f_opt:.4f}')

    for seed in range(n_seeds):
        x_p = deepcopy(x_0)
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = value(x_0, data) - f_opt if f_opt is not None else value(x_0, data)
        seed_loss_vals[seed] = loss_vals_list
        for p in range(n_comms):
            x_p_m_sum = np.zeros_like(x_p)

            for m in range(n_clients):
                x_p_m = deepcopy(x_p)
                # local run
                client_data = client_data_list[m]

                for i in range(n_local_steps):
                    batch_idx = np.random.choice(len(client_data), batch_size, replace=samp_ret)
                    input_batch = client_data[batch_idx]
                    stoch_grad = gradient(x_p_m, input_batch)
                    client_lr = calculate_clipped_lr(c_0, c_1, stoch_grad)
                    x_p_m -= client_lr * stoch_grad

                x_p_m_sum += x_p_m

            x_p = x_p_m_sum / n_clients
            loss_vals_list[p + 1] = value(x_p, data) - f_opt if f_opt is not None else value(x_p, data)
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')
        # print()

    store_result(seed_loss_vals, result_path)


def clip_fedavg(
    x_0, 
    data, 
    batch_size, 
    n_clients, 
    n_local_steps,
    n_comms,
    n_seeds,
    server_lr,
    client_lr,
    client_cl,
    cohort_size=None,
    samp_ret=True,
    f_opt=None,
):
    dim = len(x_0)
    results_dir = f'results/fl_fourth_order/dim_{dim}/x_0_{x_0[0]}/tau_{n_local_steps}/bs_{batch_size}'
    if not samp_ret:
        results_dir += '_no_ret'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if cohort_size is None:
        result_fn = f'clip_fedavg_cl_lr_{client_lr}_se_lr_{server_lr}_cl_cl_{client_cl}_{n_comms}'
    else:
        result_fn = f'clip_fedavg_pp_cl_lr_{client_lr}_se_lr_{server_lr}_cl_cl_{client_cl}_{n_comms}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for clip_fedavg with client_lr={client_lr}, server_lr={server_lr}, client_cl={client_cl} exist!')
        return

    seed_loss_vals = {}
    data_per_cl = len(data) // n_clients
    client_data_list = [data[i * data_per_cl : (i + 1) * data_per_cl] for i in range(n_clients)]
    for cld in client_data_list:
        assert len(cld) == data_per_cl
    client_idx = np.arange(n_clients)

    if cohort_size is None:
        cohort_size = n_clients

    for seed in range(n_seeds):
        x_p = deepcopy(x_0)
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = value(x_0, data) - f_opt if f_opt is not None else value(x_0, data)
        seed_loss_vals[seed] = loss_vals_list
        for p in range(n_comms):
            g_p = np.zeros_like(x_p)
            cohort = np.random.choice(client_idx, cohort_size, replace=False)

            for m in cohort:
                # local run
                x_p_m = deepcopy(x_p)
                client_data = client_data_list[m]

                for i in range(n_local_steps):
                    batch_idx = np.random.choice(len(client_data), batch_size, replace=samp_ret)
                    input_batch = client_data[batch_idx]
                    stoch_grad = gradient(x_p_m, input_batch)
                    if np.isnan(stoch_grad).any():
                        print(f'Divergence of clip_fedavg with client_lr={client_lr}, server_lr={server_lr}, client_cl={client_cl}, p={p}, m={m}, i={i}')
                        return
                    x_p_m -= client_lr * stoch_grad

                g_p_m = 1 / (n_local_steps) * (x_p - x_p_m)
                g_p += clip(g_p_m, client_cl)
                # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | Client [{m + 1}/{n_clients}] | f-f_opt={value(x_p, client_data) - f_opt:.4f}')

            g_p /= n_clients
            x_p -= server_lr * g_p
            f_val = value(x_p, data) - f_opt if f_opt is not None else value(x_p, data)
            if np.isnan(f_val):
                print(f'Divergence of clip_fedavg with client_lr={client_lr}, server_lr={server_lr}, client_cl={client_cl}, p={p}, m={m}')
                return
            loss_vals_list[p + 1] = f_val
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')

        # print()

    store_result(seed_loss_vals, result_path)


def crr_cli(
    x_0, 
    data, 
    batch_size, 
    n_clients, 
    n_epochs,
    n_seeds,
    server_lr,
    client_lr,
    c_0,
    c_1,
    f_opt=None,
):
    dim = len(x_0)
    results_dir = f'results/fl_fourth_order/dim_{dim}/x_0_{x_0[0]}/crr_cli/bs_{batch_size}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_fn = f'crr_cli_cl_lr_{client_lr}_se_lr_{server_lr}_c_0_{c_0}_c_1_{c_1}_{n_epochs}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for crr_cli with client_lr={client_lr}, server_lr={server_lr}, c_0={c_0}, c_1={c_1} exist!')
        return

    seed_loss_vals = {}
    data_per_cl = len(data) // n_clients
    client_data_list = [data[i * data_per_cl : (i + 1) * data_per_cl] for i in range(n_clients)]
    for cld in client_data_list:
        assert len(cld) == data_per_cl
    client_idx = np.arange(len(client_data_list))

    cohort_size = 2
    n_comms = n_clients // cohort_size
    assert n_clients % cohort_size == 0

    for seed in range(n_seeds):
        x_t = deepcopy(x_0)
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_epochs + 1)]
        loss_vals_list[0] = value(x_0, data) - f_opt if f_opt is not None else value(x_0, data)
        seed_loss_vals[seed] = loss_vals_list

        for t in range(n_epochs): # line 2
            g_t = np.zeros_like(x_t)
            x_t_r = deepcopy(x_t) # line 3
            client_idx_shuf = np.random.choice(client_idx, len(client_idx), replace=False) # line 4

            for r in range(n_comms): # line 5
                cohort = client_idx_shuf[r * cohort_size : (r + 1) * cohort_size]
                g_t_r = np.zeros_like(x_t_r)

                for m in cohort: # line 7
                    x_t_r_m = deepcopy(x_t_r) # line 6 and 8
                    g_t_r_m = np.zeros_like(x_t_r_m)
                    client_data = client_data_list[m].copy()
                    np.random.shuffle(client_data) # line 9
                    n_local_steps = int(np.ceil(len(client_data) / batch_size))
                    
                    for i in range(n_local_steps): # line 10
                        # batch_idx = np.random.choice(len(client_data), batch_size, replace=samp_ret) 
                        input_batch = client_data[i * batch_size : (i + 1) * batch_size]
                        stoch_grad = gradient(x_t_r_m, input_batch)
                        if np.isnan(stoch_grad).any():
                            print(f'Divergence of crr_cli with server_lr={server_lr}, client_lr={client_lr} c_0={c_0}, c_1={c_1} on t={t}, r={r}, m={m}, i={i}')
                            return
                        x_t_r_m -= client_lr * stoch_grad # line 11
                        g_t_r_m += stoch_grad * len(input_batch)

                    g_t_r += g_t_r_m / len(client_data)
                    # g_t_r += g_t_r_m / n_local_steps
                    # g_t_r += 1 / (server_lr * n_local_steps) * (x_t_r - x_t_r_m) # line 13
                    
                g_t_r /= cohort_size # line 15
                x_t_r -= server_lr * g_t_r # line 16
                g_t += g_t_r

            g_t /= n_comms
            global_lr = calculate_clipped_lr(c_0, c_1, g_t)
            x_t -= global_lr * g_t
            # x_t -= global_lr * (x_t - x_t_r) / (server_lr * n_comms) # line 10
            f_val = value(x_t, data) - f_opt if f_opt is not None else value(x_t, data)
            if np.isnan(f_val):
                print(f'Divergence of crr_cli with server_lr={server_lr}, client_lr={client_lr}, c_0={c_0}, c_1={c_1}')
            loss_vals_list[t + 1] = f_val

    store_result(seed_loss_vals, result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg', type=str)
    parser.add_argument('--x_0', type=float, default=100., help='starting point')
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--n_comms', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tau', type=int, default=10, help='number of client steps')
    parser.add_argument('--alpha_min', type=int, default=None, help='min client step size in log scale')
    parser.add_argument('--alpha_max', type=int, default=None, help='max client step size in log scale')
    parser.add_argument('--gamma_min', type=int, default=None, help='min server step size in log scale')
    parser.add_argument('--gamma_max', type=int, default=None, help='max server step size in log scale')
    parser.add_argument('--c_0_min', type=int, default=None, help='min c_0 in log scale, used to calculate step size')
    parser.add_argument('--c_0_max', type=int, default=None, help='max c_0 in log scale, used to calculate step size')
    parser.add_argument('--c_1_min', type=int, default=None, help='min c_1 in log scale, used to calculate step size')
    parser.add_argument('--c_1_max', type=int, default=None, help='max c_1 in log scale, used to calculate step size')
    parser.add_argument('--cl_min', type=int, default=None, help='min clipping level in log scale, used inside clip_fedavg')
    parser.add_argument('--cl_max', type=int, default=None, help='max clipping level in log scale, used inside clip_fedavg')
    parser.add_argument('--no_ret', action='store_false', help='whether to sample with return')
    parser.add_argument('--n_cpus', type=int, default=1, help='number of processes to run in parallel')
    args = parser.parse_args()
    
    dim = args.dim
    data = generate_data(dim)
    if dim == 1:
        x_opt, f_opt = optimal_value(data)
    else:
        f_opt = None
    
    x_0 = np.ones(dim, dtype=float) * args.x_0
    n_local_steps = args.tau
    batch_size = args.batch_size
    n_comms = args.n_comms
    n_clients = 10
    n_seeds = 10
    sample_no_return = args.no_ret

    alg = args.alg
    assert alg in ['l_sgd_jump', 'l_clip_sgd', 'clip_fedavg', 'l0l1_gd', 'crr_cli', 'clip_fedavg_pp']
    if alg == 'l_sgd_jump':
        assert args.alpha_min is not None and args.alpha_max is not None \
            and args.c_0_min is not None and args.c_0_max is not None \
                and args.c_1_min is not None and args.c_1_max is not None
        client_lr_list = np.logspace(
            args.alpha_min, args.alpha_max, args.alpha_max - args.alpha_min + 1
        )
        c_0_list = np.logspace(
            args.c_0_min, args.c_0_max, args.c_0_max - args.c_0_min + 1
        )
        c_1_list = np.logspace(
            args.c_1_min, args.c_1_max, args.c_1_max - args.c_1_min + 1
        )
        n_cpus = args.n_cpus
        partial_run = partial(
            local_sgd_jump,
            x_0,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            samp_ret=sample_no_return,
            f_opt=f_opt,
        )
        args_product = product(client_lr_list, c_0_list, c_1_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for client_lr, c_0, c_1 in args_product:
                partial_run(client_lr, c_0, c_1)

    elif alg == 'l0l1_gd':
        assert args.c_0_min is not None and args.c_0_max is not None \
            and args.c_1_min is not None and args.c_1_max is not None
        c_0_list = np.logspace(
            args.c_0_min, args.c_0_max, args.c_0_max - args.c_0_min + 1
        )
        c_1_list = np.logspace(
            args.c_1_min, args.c_1_max, args.c_1_max - args.c_1_min + 1
        )
        n_cpus = args.n_cpus
        partial_run = partial(
            l0l1_gd,
            x_0,
            data, 
            n_comms,
            n_seeds,
            f_opt=f_opt,
        )
        args_product = product(c_0_list, c_1_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for c_0, c_1 in args_product:
                partial_run(c_0, c_1)
             
    elif alg == 'l_clip_sgd':
        assert args.c_0_min is not None and args.c_0_max is not None \
            and args.c_1_min is not None and args.c_1_max is not None
        c_0_list = np.logspace(
            args.c_0_min, args.c_0_max, args.c_0_max - args.c_0_min + 1
        )
        c_1_list = np.logspace(
            args.c_1_min, args.c_1_max, args.c_1_max - args.c_1_min + 1
        )
        n_cpus = args.n_cpus
        partial_run = partial(
            local_clip_sgd,
            x_0,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            samp_ret=sample_no_return,
            f_opt=f_opt,
        )
        args_product = product(c_0_list, c_1_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for c_0, c_1 in args_product:
                partial_run(c_0, c_1)
       
    elif alg == 'clip_fedavg':
        assert args.alpha_min is not None and args.alpha_max is not None \
            and args.gamma_min is not None and args.gamma_max is not None \
                and args.cl_min is not None and args.cl_max is not None
        client_lr_list = np.logspace(
            args.alpha_min, args.alpha_max, args.alpha_max - args.alpha_min + 1
        )
        server_lr_list = np.logspace(
            args.gamma_min, args.gamma_max, args.gamma_max - args.gamma_min + 1
        )
        client_clip_level_list = np.logspace(
            args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1
        )
        n_cpus = args.n_cpus
        partial_run = partial(
            clip_fedavg,
            x_0,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            samp_ret=sample_no_return,
            f_opt=f_opt,
        )
        args_product = product(server_lr_list, client_lr_list, client_clip_level_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for server_lr, client_lr, client_clip_level in args_product:
                partial_run(server_lr, client_lr, client_clip_level)

    elif alg == 'crr_cli':
        assert args.alpha_min is not None and args.alpha_max is not None \
            and args.gamma_min is not None and args.gamma_max is not None \
                and args.c_0_min is not None and args.c_0_max is not None \
                    and args.c_1_min is not None and args.c_1_max is not None

        client_lr_list = np.logspace(
            args.alpha_min, args.alpha_max, args.alpha_max - args.alpha_min + 1
        )
        server_lr_list = np.logspace(
            args.gamma_min, args.gamma_max, args.gamma_max - args.gamma_min + 1
        )
        c_0_list = np.logspace(
            args.c_0_min, args.c_0_max, args.c_0_max - args.c_0_min + 1
        )
        c_1_list = np.logspace(
            args.c_1_min, args.c_1_max, args.c_1_max - args.c_1_min + 1
        )
        n_cpus = args.n_cpus
        n_epochs = n_comms
        partial_run = partial(
            crr_cli,
            x_0,
            data,
            batch_size,
            n_clients,
            n_comms,
            n_seeds,
            f_opt=f_opt,
        )
        args_product = product(server_lr_list, client_lr_list, c_0_list, c_1_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for server_lr, client_lr, c_0, c_1 in args_product:
                partial_run(server_lr, client_lr, c_0, c_1)

    elif alg == 'clip_fedavg_pp':
        assert args.alpha_min is not None and args.alpha_max is not None \
            and args.gamma_min is not None and args.gamma_max is not None \
                and args.cl_min is not None and args.cl_max is not None
        client_lr_list = np.logspace(
            args.alpha_min, args.alpha_max, args.alpha_max - args.alpha_min + 1
        )
        server_lr_list = np.logspace(
            args.gamma_min, args.gamma_max, args.gamma_max - args.gamma_min + 1
        )
        client_clip_level_list = np.logspace(
            args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1
        )
        n_cpus = args.n_cpus
        partial_run = partial(
            clip_fedavg,
            x_0,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            cohort_size=2,
            samp_ret=sample_no_return,
            f_opt=f_opt,
        )
        args_product = product(server_lr_list, client_lr_list, client_clip_level_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for server_lr, client_lr, client_clip_level in args_product:
                partial_run(server_lr, client_lr, client_clip_level)
