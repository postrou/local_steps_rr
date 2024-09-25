import os
import argparse
from multiprocessing import Pool
import pickle
from functools import partial
from itertools import product

import numpy as np


def generate_data():
    np.random.seed(0)
    data = np.random.uniform(-10, 10, 1000)
    data.sort()
    for i in range(len(data) - 1):
        assert data[i] < data[i + 1]
    return data

    
def store_result(result, result_path):
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)


def value(x, data_batch):
    return np.mean([(x - x_0) ** 4 for x_0 in data_batch])
    

def gradient(x, data_batch):
    return 4 * np.mean([(x - x_0) ** 3 for x_0 in data_batch])


def hessian(x, data_batch):
    return 12 * np.mean([(x - x_0) ** 2 for x_0 in data_batch])


def optimal_value(data):
    step_size = 1

    x0 = 1000
    x = x0
    N = 20
    for i in range(N):
        grad = gradient(x, data)
        hess = hessian(x, data)
        x -= step_size * 1 / hess * grad
        f_value = value(x, data)

    x_opt = x
    f_opt = f_value
    return x_opt, f_opt


def calculate_clipped_lr(c_0, c_1, g_p):
    return 1 / (c_0 + c_1 * abs(g_p))


def clip(vector, clip_level):
    return vector * min(1, clip_level / np.linalg.norm(vector))

    
def l0l1_gd(
    x_0, 
    f_opt,
    data, 
    n_comms,
    n_seeds,
    c_0,
    c_1,
):
    results_dir = f'results/fl_fourth_order/x_0_{x_0}/gd'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_fn = f'gd_c_0_{c_0}_c_1_{c_1}_{n_comms}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for l0l1_gd with c_0={c_0}, c_1={c_1} exist!')
        print(result_path)
        return

    seed_loss_vals = {}
    # print(f'Number of data points per client: {data_per_par}')
    # print(f'Client data lens: {[len(d) for d in client_data_list]}')
    # print(f'Comm [{0}/{n_comms}] f-f_opt={value(x_p, data) - f_opt:.4f}')

    for seed in range(n_seeds):
        x_p = x_0
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = abs(value(x_0, data) - f_opt)
        seed_loss_vals[seed] = loss_vals_list

        for p in range(n_comms):
            g_p = gradient(x_p, data)
            server_lr = calculate_clipped_lr(c_0, c_1, g_p)
            x_p -= server_lr * g_p
            f_val = abs(value(x_p, data) - f_opt)
            if np.isnan(f_val):
                print(f'Divergence of l0l1_gd with c_0={c_0}, c_1={c_1}')
                return
            loss_vals_list[p + 1] = f_val
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')

        # print()

    store_result(seed_loss_vals, result_path)


def local_sgd_jump(
    x_0, 
    f_opt,
    data, 
    batch_size, 
    n_clients, 
    n_local_steps,
    n_comms,
    n_seeds,
    client_lr, 
    c_0,
    c_1,
    samp_ret=True
):
    results_dir = f'results/fl_fourth_order/x_0_{x_0}/tau_{n_local_steps}/bs_{batch_size}'
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
        x_p = x_0
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = abs(value(x_0, data) - f_opt)
        seed_loss_vals[seed] = loss_vals_list

        for p in range(n_comms):
            g_p = 0

            for m in range(n_clients):
                # local run
                x_p_m = x_p
                client_data = client_data_list[m]

                for i in range(n_local_steps):
                    input_batch = np.random.choice(client_data, batch_size, replace=samp_ret)
                    stoch_grad = gradient(x_p_m, input_batch)
                    if np.isnan(stoch_grad):
                        print(f'Divergence of local_sgd_jump with client_lr={client_lr}, c_0={c_0}, c_1={c_1} on p={p}, m={m}, i={i}')
                        return
                    x_p_m -= client_lr * stoch_grad

                g_p += x_p - x_p_m

            g_p *= 1 / (client_lr * n_clients * n_local_steps)
            server_lr = calculate_clipped_lr(c_0, c_1, g_p)
            x_p -= server_lr * g_p
            f_val = abs(value(x_p, data) - f_opt)
            if np.isnan(f_val):
                print(f'Divergence of local_sgd_jump with client_lr={client_lr}, c_0={c_0}, c_1={c_1}')
                return
            loss_vals_list[p + 1] = f_val
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')

        # print()

    store_result(seed_loss_vals, result_path)


def local_clip_sgd(
    x_0, 
    f_opt,
    data, 
    batch_size, 
    n_clients, 
    n_local_steps,
    n_comms,
    n_seeds,
    c_0,
    c_1,
    samp_ret=True
):
    results_dir = f'results/fl_fourth_order/x_0_{x_0}/tau_{n_local_steps}/bs_{batch_size}'
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
        x_p = x_0
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = abs(value(x_0, data) - f_opt)
        seed_loss_vals[seed] = loss_vals_list
        for p in range(n_comms):
            x_p_m_sum = 0

            for m in range(n_clients):
                x_p_m = x_p
                # local run
                client_data = client_data_list[m]

                for i in range(n_local_steps):
                    input_batch = np.random.choice(client_data, batch_size, replace=samp_ret)
                    stoch_grad = gradient(x_p_m, input_batch)
                    client_lr = calculate_clipped_lr(c_0, c_1, stoch_grad)
                    x_p_m -= client_lr * stoch_grad

                x_p_m_sum += x_p_m

            x_p = x_p_m_sum / n_clients
            loss_vals_list[p + 1] = abs(value(x_p, data) - f_opt)
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')
        # print()

    store_result(seed_loss_vals, result_path)

    
def clip_fedavg(
    x_0, 
    f_opt,
    data, 
    batch_size, 
    n_clients, 
    n_local_steps,
    n_comms,
    n_seeds,
    server_lr,
    client_lr,
    client_cl,
    samp_ret=True
):
    results_dir = f'results/fl_fourth_order/x_0_{x_0}/tau_{n_local_steps}/bs_{batch_size}'
    if not samp_ret:
        results_dir += '_no_ret'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_fn = f'clip_fedavg_cl_lr_{client_lr}_se_lr_{server_lr}_cl_cl_{client_cl}_{n_comms}'
    result_path = os.path.join(results_dir, result_fn)
    if os.path.exists(result_path):
        print(f'Results for clip_fedavg with client_lr={client_lr}, server_lr={server_lr}, client_cl={client_cl} exist!')
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
        x_p = x_0
        np.random.seed(seed)
        loss_vals_list = [0.0 for _ in range(n_comms + 1)]
        loss_vals_list[0] = abs(value(x_0, data) - f_opt)
        seed_loss_vals[seed] = loss_vals_list
        for p in range(n_comms):
            g_p = 0

            for m in range(n_clients):
                # local run
                x_p_m = x_p
                client_data = client_data_list[m]

                for i in range(n_local_steps):
                    input_batch = np.random.choice(client_data, batch_size, replace=samp_ret)
                    stoch_grad = gradient(x_p_m, input_batch)
                    if np.isnan(stoch_grad):
                        print(f'Divergence of clip_fedavg with client_lr={client_lr}, server_lr={server_lr}, client_cl={client_cl}')
                        return
                    x_p_m -= client_lr * stoch_grad

                g_p_m = 1 / (client_lr * n_local_steps) * (x_p - x_p_m)
                g_p += clip(g_p_m, client_cl)
                # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | Client [{m + 1}/{n_clients}] | f-f_opt={value(x_p, client_data) - f_opt:.4f}')

            g_p /= n_clients
            x_p -= server_lr * g_p
            f_val = abs(value(x_p, data) - f_opt)
            if np.isnan(f_val):
                print(f'Divergence of local_sgd_jump with client_lr={client_lr}, server_lr={server_lr}, client_cl={client_cl}')
                return
            loss_vals_list[p + 1] = f_val
            # print(f'Seed [{seed+1}/{n_seeds}] | Comm [{p+1}/{n_comms}] | f-f_opt={value(x_p, data) - f_opt:.4f}')

        # print()

    store_result(seed_loss_vals, result_path)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg', type=str)
    parser.add_argument('--x_0', type=float, default=100., help='starting point')
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
    
    data = generate_data()
    x_opt, f_opt = optimal_value(data)
    
    x_0 = args.x_0
    n_local_steps = args.tau
    batch_size = args.batch_size
    n_comms = args.n_comms
    n_clients = 10
    n_seeds = 10
    sample_no_return = args.no_ret

    alg = args.alg
    assert alg in ['l_sgd_jump', 'l_clip_sgd', 'clip_fedavg', 'l0l1_gd']
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
            f_opt,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            samp_ret=sample_no_return
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
            f_opt,
            data, 
            n_comms,
            n_seeds
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
            f_opt,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            samp_ret=sample_no_return
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
            f_opt,
            data,
            batch_size,
            n_clients,
            n_local_steps,
            n_comms,
            n_seeds,
            samp_ret=sample_no_return
        )
        args_product = product(server_lr_list, client_lr_list, client_clip_level_list)
        if n_cpus > 1:
            pool = Pool(n_cpus)
            pool.starmap(partial_run, args_product)
        else:
            for server_lr, client_lr, client_clip_level in args_product:
                partial_run(server_lr, client_lr, client_clip_level)
