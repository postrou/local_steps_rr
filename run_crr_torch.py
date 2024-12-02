import os
import argparse
import random
from multiprocessing import Pool
from functools import partial
from itertools import product
import pickle
import sys

import torch
import numpy as np
from tqdm.auto import tqdm

from src.optimizers_torch import ClippedSGD, NASTYA, ClERR, ClERRHeuristic
from src.loss_functions.models import build_lenet_model, build_resnet_model
from src.utils_torch import *


def redirect_output(log_file):
    sys.stdout = open(log_file, "w")
    sys.stderr = open(log_file, "w")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def compute_and_store_epoch_results(
    task,
    epoch,
    train_loss_vals,
    train_gns,
    train_acc_vals,
    test_loss_vals,
    test_gns,
    test_acc_vals,
    train_data,
    test_data,
    model,
    criterion,
    device,
):
    if task == "cifar10":
        train_loss, train_acc, train_gn, test_loss, test_acc, test_gn = (
            cifar_epoch_result(train_data, test_data, model, criterion, device)
        )
    elif task.startswith('logreg'):
        train_loss, train_acc, train_gn, test_loss, test_acc, test_gn = (
            logreg_epoch_result(train_data, test_data, model, criterion, device)
        )
    elif task == "penn":
        train_loss, train_gn, test_loss, test_gn = penn_epoch_result(
            train_data, test_data, model, criterion, device
        )
    train_loss_vals[epoch + 1] = train_loss
    train_gns[epoch + 1] = train_gn
    test_loss_vals[epoch + 1] = test_loss
    test_gns[epoch + 1] = test_gn
    if task == 'cifar10' or task.startswith('logreg'):
        train_acc_vals[epoch + 1] = train_acc
        test_acc_vals[epoch + 1] = test_acc


def init_seed(seed, n_epochs):
    set_seed(seed)
    train_loss_vals = [0.0] * (n_epochs + 1)
    train_gns = [0.0] * (n_epochs + 1)
    train_acc_vals = [0.0] * (n_epochs + 1)
    train_local_loss_vals = []
    train_local_gns = []
    train_local_acc_vals = []
    test_loss_vals = [0.0] * (n_epochs + 1)
    test_gns = [0.0] * (n_epochs + 1)
    test_acc_vals = [0.0] * (n_epochs + 1)
    return (
        train_loss_vals,
        train_gns,
        train_acc_vals,
        train_local_loss_vals,
        train_local_gns,
        train_local_acc_vals,
        test_loss_vals,
        test_gns,
        test_acc_vals,
    )


def init_optimizer(
    alg, parameters, n_batches, lr, cl, inner_lr, inner_cl, c_0=None, c_1=None
):
    if alg == "so":
        optimizer = torch.optim.SGD(parameters, lr=lr)
    elif alg == "cso":
        optimizer = ClippedSGD(params=parameters, clip_level=cl, lr=lr)
    elif alg == "nastya":
        optimizer = NASTYA(
            params=parameters,
            n_batches=n_batches,
            lr=inner_lr,
            outer_lr=lr,
        )
    elif alg == "clerr":
        if c_0 is None:
            c_0 = 1 / (2 * lr)
        if c_1 is None:
            c_1 = c_0 / cl
        optimizer = ClERR(
            params=parameters,
            c_0=c_0,
            c_1=c_1,
            n_batches=n_batches,
            lr=inner_lr,
            use_g_in_outer_step=True,
        )
    elif alg == "clerr_heuristic":
        if c_0 is None:
            c_0 = 1 / (2 * lr)
        if c_1 is None:
            c_1 = c_0 / cl
        optimizer = ClERRHeuristic(
            params=parameters,
            c_0=c_0,
            c_1=c_1,
            in_clip_level=inner_cl,
            lr=inner_lr,
        )
    return optimizer


def store_seed_results(
    seed,
    train_loss_vals,
    train_gns,
    train_acc_vals,
    train_local_loss_vals,
    train_local_gns,
    train_local_acc_vals,
    test_loss_vals,
    test_gns,
    test_acc_vals,
    train_loss_vals_all,
    train_gns_all,
    train_acc_vals_all,
    train_local_loss_vals_all,
    train_local_gns_all,
    train_local_acc_vals_all,
    test_loss_vals_all,
    test_gns_all,
    test_acc_vals_all,
):
    train_loss_vals_all[seed] = train_loss_vals.copy()
    train_acc_vals_all[seed] = train_acc_vals.copy()
    train_gns_all[seed] = train_gns.copy()
    train_local_loss_vals_all[seed] = train_local_loss_vals.copy()
    train_local_acc_vals_all[seed] = train_local_acc_vals.copy()
    train_local_gns_all[seed] = train_local_gns.copy()
    test_loss_vals_all[seed] = test_loss_vals.copy()
    test_acc_vals_all[seed] = test_acc_vals.copy()
    test_gns_all[seed] = test_gns.copy()


def print_train_test_results(
    alg,
    seed,
    n_seeds,
    epoch,
    n_epochs,
    train_loss,
    train_gn,
    test_loss,
    test_gn,
    lr,
    cl=None,
    inner_lr=None,
    inner_cl=None,
    c_0=None,
    c_1=None,
    train_acc=None,
    test_acc=None,
):
    if alg == "so":
        train_str = f"💪💪💪 SO | LR={lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Train loss={train_loss:.4f} | Train gn={train_gn:.4f}"
        test_str = f"🧪🧪🧪 SO | LR={lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Test loss={test_loss:.4f} | Test gn={test_gn:.4f}"
    elif alg == "cso":
        train_str = f"💪💪💪 CSO | CL={cl} | LR={lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Train loss={train_loss:.4f} | Train gn={train_gn:.4f}"
        test_str = f"🧪🧪🧪 CSO | CL={cl} | LR={lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Test loss={test_loss:.4f} | Test gn={test_gn:.4f}"
    elif alg == "nastya":
        train_str = f"💪💪💪 NASTYA-SO | LR={lr} | Inner LR={inner_lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Train loss={train_loss:.4f} | Train gn={train_gn:.4f}"
        test_str = f"🧪🧪🧪 NASTYA-SO | LR={lr} | Inner LR={inner_lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Test loss={test_loss:.4f} | Test gn={test_gn:.4f}"
    elif alg == "clerr":
        train_str = f"💪💪💪 ClERR-g-SO | CL={cl} | LR={lr} | Inner LR={inner_lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Train loss={train_loss:.4f} | Train gn={train_gn:.4f}"
        test_str = f"🧪🧪🧪 ClERR-g-SO | CL={cl} | LR={lr} | Inner LR={inner_lr} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Test loss={test_loss:.4f} | Test gn={test_gn:.4f}"
    elif alg == "clerr_heuristic":
        if c_0 is None and c_1 is None:
            train_str = f"💪💪💪 ClERR-heuristic | CL={cl} | LR={lr} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Train loss={train_loss:.4f} | Train gn={train_gn:.4f}"
            test_str = f"🧪🧪🧪 ClERR-heuristic | CL={cl} | LR={lr} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Test loss={test_loss:.4f} | Test gn={test_gn:.4f}"
        else:
            train_str = f"💪💪💪 ClERR-heuristic | c_0={c_0} | c_1={c_1} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Train loss={train_loss:.4f} | Train gn={train_gn:.4f}"
            test_str = f"🧪🧪🧪 ClERR-heuristic | c_0={c_0} | c_1={c_1} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed}/{n_seeds} | Epoch {epoch}/{n_epochs} | Test loss={test_loss:.4f} | Test gn={test_gn:.4f}"

    if train_acc is not None and test_acc is not None:
        train_str += f" | Train acc={train_acc:.4f}"
        test_str += f" | Test acc={test_acc:.4f}"
    print(train_str)
    print(test_str)


def train_shuffling(
    test,
    task,
    model_type,
    add_het,
    alg,
    n_seeds,
    n_epochs,
    batch_size,
    lr,
    cl=None,
    inner_lr=None,
    inner_cl=None,
    c_0=None,
    c_1=None,
):
    assert task in ["cifar10", "penn", 'logreg_covtype', 'logreg_gisette', 'logreg_realsim', 'logreg_mushrooms'], f"Task {task} is not implemented!"
    if not test:
        # all stdout goes to log file
        log_dir = f"logs/{task}/{model_type}/bs_{batch_size}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if alg == "so":
            log_fn = f"so_lr_{lr}_seeds_{n_seeds}_{n_epochs}.log"
        elif alg == "cso":
            log_fn = f"c_{cl}_lr_{lr}_so_seeds_{n_seeds}_{n_epochs}.log"
        elif alg == "nastya":
            log_fn = (
                f"nastya_lr_{lr}_in_lr_{inner_lr}_so_seeds_{n_seeds}_{n_epochs}.log"
            )
        elif alg == "clerr":
            log_fn = f"clerr_g_c_{cl}_lr_{lr}_in_lr_{inner_lr}_so_seeds_{n_seeds}_{n_epochs}.log"
        elif alg == "clerr_heuristic":
            if c_0 is None and c_1 is None:
                log_fn = f"clerr_heuristic_c_{cl}_lr_{lr}_in_lr_{inner_lr}_so_seeds_{n_seeds}_{n_epochs}.log"
            else:
                log_fn = f"clerr_heuristic_c_0_{c_0}_c_1_{c_1}_in_lr_{inner_lr}_in_cl_{inner_cl}_so_seeds_{n_seeds}_{n_epochs}.log"
        log_path = os.path.join(log_dir, log_fn)
        redirect_output(log_path)

    task_dir = task + "_het" if add_het else task
    results_dir = f"results/{task_dir}/{model_type}/bs_{batch_size}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if alg == "so":
        print(f"Starting {alg} for lr={lr}")
        results_fn = f"so_lr_{lr}_seeds_{n_seeds}_{n_epochs}"
    elif alg == "cso":
        print(f"Starting {alg} for cl={cl}, lr={lr}")
        results_fn = f"c_{cl}_lr_{lr}_so_seeds_{n_seeds}_{n_epochs}"
    elif alg == "nastya":
        print(f"Starting {alg} for lr={lr}, inner_lr={inner_lr}")
        results_fn = f"nastya_lr_{lr}_in_lr_{inner_lr}_so_seeds_{n_seeds}_{n_epochs}"
    elif alg == "clerr":
        print(f"Starting {alg}-g for cl={cl}, lr={lr}, inner_lr={inner_lr}")
        results_fn = (
            f"clerr_g_c_{cl}_lr_{lr}_in_lr_{inner_lr}_so_seeds_{n_seeds}_{n_epochs}"
        )
    elif alg == "clerr_heuristic":
        assert inner_cl is not None and inner_lr is not None
        if c_0 is None or c_1 is None:
            print(
                f"Starting {alg} for cl={cl}, lr={lr}, inner_lr={inner_lr}, inner_cl={inner_cl}"
            )
            results_fn = f"clerr_heuristic_c_{cl}_lr_{lr}_in_lr_{inner_lr}_in_cl_{inner_cl}_so_seeds_{n_seeds}_{n_epochs}"
        else:
            print(f"Starting {alg} for c_0={c_0}, c_1={c_1}, inner_lr={inner_lr}, inner_cl={inner_cl}")
            results_fn = f"clerr_heuristic_c_0_{c_0}_c_1_{c_1}_in_lr_{inner_lr}_in_cl_{inner_cl}_so_seeds_{n_seeds}_{n_epochs}"
    results_path = os.path.join(results_dir, results_fn)

    if os.path.exists(results_path) and not test:
        if alg == "so":
            print(
                f"Results for so, run for {n_seeds} seeds, {n_epochs} epochs with lr={lr} already exist!"
            )
        elif alg == "cso":
            print(
                f"Results for cso, run for {n_seeds} seeds, {n_epochs} epochs with cl={cl}, lr={lr} already exist!"
            )
        elif alg == "nastya":
            print(
                f"Results for nastya-so, run for {n_seeds} seeds, {n_epochs} epochs with lr={lr}, inner_lr={inner_lr} already exist!"
            )
        elif alg == "nastya":
            print(
                f"Results for nastya-so, run for {n_seeds} seeds, {n_epochs} epochs with lr={lr}, inner_lr={inner_lr} already exist!"
            )
        elif alg == "clerr":
            print(
                f"Results for clerr-g-so, run for {n_seeds} seeds, {n_epochs} epochs with cl={cl},  lr={lr}, inner_lr={inner_lr} already exist!"
            )
        elif alg == "clerr_heuristic":
            if c_0 is None or c_1 is None:
                print(
                    f"Results for clerr-heuristic, run for {n_seeds} seeds, {n_epochs} epochs with c_0={c_0}, c_1={c_1}, inner_lr={inner_lr}, inner_cl={inner_cl} already exist!"
                )
            else:
                print(
                    f"Results for clerr-heuristic, run for {n_seeds} seeds, {n_epochs} epochs with cl={cl}, lr={lr}, inner_lr={inner_lr}, inner_cl={inner_cl} already exist!"
                )
        return

    device = "cuda"
    train_loss_vals_all = {}
    train_acc_vals_all = {}
    train_gns_all = {}
    train_local_loss_vals_all = {}
    train_local_acc_vals_all = {}
    train_local_gns_all = {}
    test_loss_vals_all = {}
    test_acc_vals_all = {}
    test_gns_all = {}

    for seed in range(n_seeds):
        (
            train_loss_vals,
            train_gns,
            train_acc_vals,
            train_local_loss_vals,
            train_local_gns,
            train_local_acc_vals,
            test_loss_vals,
            test_gns,
            test_acc_vals,
        ) = init_seed(seed, n_epochs)

        if task == "cifar10":
            train_loader, train_data, test_data = cifar_load_data("data/cifar10/", batch_size, add_het, model_type)
        elif task == "penn":
            n_tokens, train_loader, test_data = penn_load_data(
                "data/penn/", batch_size
            )
        elif task == 'logreg_covtype':
            train_loader, train_data, test_data = logreg_load_data('data/covtype.bz2', batch_size)
        elif task == 'logreg_gisette':
            train_loader, train_data, test_data = logreg_load_data('data/gisette.bz2', batch_size)
        elif task == 'logreg_realsim':
            train_loader, train_data, test_data = logreg_load_data('data/real-sim.bz2', batch_size)
        elif task == 'logreg_mushrooms':
            train_loader, train_data, test_data = logreg_load_data('data/mushrooms', batch_size)

        if model_type == "resnet":
            model, criterion = build_resnet_model(device)
        elif model_type == 'lenet':
            model, criterion = build_lenet_model(device)
        elif model_type == 'linear':
            input_dim = train_data[0][0].shape[1]
            output_dim = 2
            model, criterion = build_linear_model(input_dim, output_dim, device)
        elif task == "penn":
            model, criterion = build_lstm_model(n_tokens, device)
        model.train()
        optimizer = init_optimizer(
            alg, model.parameters(), len(train_loader), lr, cl, inner_lr, inner_cl, c_0, c_1
        )

        # print('Computing and storing initial results...')
        compute_and_store_epoch_results(
            task,
            -1,
            train_loss_vals,
            train_gns,
            train_acc_vals,
            test_loss_vals,
            test_gns,
            test_acc_vals,
            train_data,
            test_data,
            model,
            criterion,
            device,
        )

        train_acc = train_acc_vals[0] if task == "cifar10" else None
        test_acc = test_acc_vals[0] if task == "cifar10" else None
        print_train_test_results(
            alg,
            seed + 1,
            n_seeds,
            0,
            n_epochs,
            train_loss_vals[0],
            train_gns[0],
            test_loss_vals[0],
            test_gns[0],
            lr,
            cl,
            inner_lr,
            inner_cl,
            c_0,
            c_1,
            train_acc,
            test_acc,
        )

        initial_loss, initial_gn, initial_acc = None, None, None
        for epoch in range(n_epochs):
            if alg == "so":
                progress_bar = tqdm(
                    train_loader,
                    desc=f"🤖🤖🤖 SO | LR={lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}",
                    leave=True,
                )
            elif alg == "cso":
                progress_bar = tqdm(
                    train_loader,
                    desc=f"🤖🤖🤖 CSO | CL={cl} | LR={lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}",
                    leave=True,
                )
            elif alg == "nastya":
                x_start_epoch = [
                    p.detach().requires_grad_(False) for p in model.parameters()
                ]
                progress_bar = tqdm(
                    train_loader,
                    desc=f"🤖🤖🤖 NASTYA-SO | LR={lr} | Inner LR={inner_lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}",
                    leave=True,
                )
            elif alg == "clerr":
                x_start_epoch = [
                    p.detach().requires_grad_(False) for p in model.parameters()
                ]
                progress_bar = tqdm(
                    train_loader,
                    desc=f"🤖🤖🤖 ClERR-g-SO | CL={cl} | LR={lr} | Inner LR={inner_lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}",
                    leave=True,
                )
            elif alg == 'clerr_heuristic':
                x_start_epoch = [
                    p.detach().requires_grad_(False) for p in model.parameters()
                ]
                if c_0 is None and c_1 is None:
                    progress_bar = tqdm(
                        train_loader,
                        desc=f"🤖🤖🤖 ClERR-heuristic | CL={cl} | LR={lr} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}",
                        leave=True,
                    )
                else:
                    progress_bar = tqdm(
                        train_loader,
                        desc=f"🤖🤖🤖 ClERR-heuristic | c_0={c_0} | c_1={c_1} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}",
                        leave=True,
                    )
 
            # progress_bar = train_loader

            if task == "cifar10":
                cifar_train_epoch(
                    progress_bar,
                    model,
                    criterion,
                    optimizer,
                    train_local_loss_vals,
                    train_local_gns,
                    train_local_acc_vals,
                    initial_loss,
                    initial_gn,
                    initial_acc,
                    device,
                )

            if task.startswith("logreg"):
                logreg_train_epoch(
                    progress_bar,
                    model,
                    criterion,
                    optimizer,
                    train_local_loss_vals,
                    train_local_gns,
                    train_local_acc_vals,
                    initial_loss,
                    initial_gn,
                    initial_acc,
                    device,
                )
            elif task == "penn":
                penn_train_epoch(
                    progress_bar,
                    model,
                    criterion,
                    optimizer,
                    batch_size,
                    train_local_loss_vals,
                    train_local_gns,
                    initial_loss,
                    initial_gn,
                    device,
                )

            if type(optimizer) in [ClERR, NASTYA, ClERRHeuristic]:
                optimizer.outer_step(x_start_epoch)
                optimizer.init_g()

            compute_and_store_epoch_results(
                task,
                epoch,
                train_loss_vals,
                train_gns,
                train_acc_vals,
                test_loss_vals,
                test_gns,
                test_acc_vals,
                train_data,
                test_data,
                model,
                criterion,
                device,
            )

            train_acc = train_acc_vals[epoch + 1] if task == 'cifar10' or task.startswith('logreg') else None
            test_acc = test_acc_vals[epoch + 1] if task == 'cifar10' or task.startswith('logreg') else None
            print_train_test_results(
                alg,
                seed + 1,
                n_seeds,
                epoch + 1,
                n_epochs,
                train_loss_vals[epoch + 1],
                train_gns[epoch + 1],
                test_loss_vals[epoch + 1],
                test_gns[epoch + 1],
                lr,
                cl,
                inner_lr,
                inner_cl,
                c_0,
                c_1,
                train_acc,
                test_acc,
            )

        if alg == "so":
            print(f"🌱🌱🌱 SO | LR={lr} | Seed {seed + 1}/{n_seeds} finished!")
        elif alg == "cso":
            print(
                f"🌱🌱🌱 CSO | CL={cl} | LR={lr} | Seed {seed + 1}/{n_seeds} finished!"
            )
        elif alg == "nastya":
            print(
                f"🌱🌱🌱 NASTYA-SO | LR={lr} | Inner LR={inner_lr} | Seed {seed + 1}/{n_seeds} finished!"
            )
        elif alg == "clerr":
            print(
                f"🌱🌱🌱 ClERR-g-SO | CL={cl} | LR={lr} | Inner LR={inner_lr} | Seed {seed + 1}/{n_seeds} finished!"
            )
        elif alg == "clerr_heuristic":
            if c_0 is None and c_1 is None:
                print(
                    f"🌱🌱🌱 ClERR-heuristic | CL={cl} | LR={lr} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed + 1}/{n_seeds} finished!"
                )
            else:
                print(
                    f"🌱🌱🌱 ClERR-heuristic | c_0={c_0} | c_1={c_1} | Inner LR={inner_lr} | Inner CL={inner_cl} | Seed {seed + 1}/{n_seeds} finished!"
                )
        store_seed_results(
            seed,
            train_loss_vals,
            train_gns,
            train_acc_vals,
            train_local_loss_vals,
            train_local_gns,
            train_local_acc_vals,
            test_loss_vals,
            test_gns,
            test_acc_vals,
            train_loss_vals_all,
            train_gns_all,
            train_acc_vals_all,
            train_local_loss_vals_all,
            train_local_gns_all,
            train_local_acc_vals_all,
            test_loss_vals_all,
            test_gns_all,
            test_acc_vals_all,
        )

    result = {
        "train_loss": train_loss_vals_all,
        "train_gn": train_gns_all,
        "train_local_loss": train_local_loss_vals_all,
        "train_local_gn": train_local_gns_all,
        "train_local_acc": train_local_acc_vals_all,
        "test_loss": test_loss_vals_all,
        "test_gn": test_gns_all,
    }
    if task == 'cifar10' or task.startswith('logreg'):
        result["train_acc"] = train_acc_vals_all
        result["test_acc"] = test_acc_vals_all

    with open(results_path, "wb") as f:
        pickle.dump(result, f)


def get_argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", type=str)
    parser.add_argument("task", type=str)
    parser.add_argument("--model", type=str, default='resnet')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--cl_min", type=int, default=None, help="min clip level in log scale"
    )
    parser.add_argument(
        "--cl_max", type=int, default=None, help="max clip level in log scale"
    )
    parser.add_argument(
        "--lr_min",
        type=int,
        default=None,
        help="min step size in log scale (used for outer lr computation in NASTYA and ClERR)",
    )
    parser.add_argument(
        "--lr_max",
        type=int,
        default=None,
        help="max step size in log scale (used for outer lr computation in NASTYA and ClERR)",
    )
    parser.add_argument(
        "--in_lr_min",
        type=int,
        default=None,
        help="min inner step size in log scale (for CLERR)",
    )
    parser.add_argument(
        "--in_lr_max",
        type=int,
        default=None,
        help="max inner step size in log scale (for CLERR)",
    )
    parser.add_argument(
        "--in_cl",
        type=int,
        default=None,
        help="clip level for inner step size for ClERR-Heuristic algorithm",
    )
    parser.add_argument(
        "--c_0_min",
        type=int,
        default=None,
        help="min of constant c_0 in log scale, used in calculation of outer step size in ClERR and ClERR-Heuristic (now implemented only for ClERR-Heuristic)",
    )
    parser.add_argument(
        "--c_0_max",
        type=int,
        default=None,
        help="max constant c_0 in log scale, used in calculation of outer step size in ClERR and ClERR-Heuristic (now implemented only for ClERR-Heuristic)",
    )
    parser.add_argument(
        "--c_1_min",
        type=int,
        default=None,
        help="min constant c_1 in log scale, used in calculation of outer step size in ClERR and ClERR-Heuristic (now implemented only for ClERR-Heuristic)",
    )
    parser.add_argument(
        "--c_1_max",
        type=int,
        default=None,
        help="max constant c_1 in log scale, used in calculation of outer step size in ClERR and ClERR-Heuristic (now implemented only for ClERR-Heuristic)",
    )
    parser.add_argument(
        '--add_het',
        action='store_true'
    )
    parser.add_argument(
        "--n_cpus", type=int, default=1, help="number of processes to run in parallel"
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="number of cuda to use (default -1 is for cpu)",
    )
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_argparse_args()

    alg = args.alg
    task = args.task
    model_type = args.model
    assert alg in ["so", "cso", "nastya", "clerr", "clerr_heuristic"]
    assert task in ["cifar10", "penn", 'logreg_covtype', 'logreg_gisette', 'logreg_realsim', 'logreg_mushrooms']
    assert model_type in ['resnet', 'lenet', 'lstm', 'linear']

    if alg != 'clerr_heuristic':
        assert (
            args.lr_min is not None and args.lr_max is not None
        ), f"You did not provide --lr_min or --lr_max for algorithm {alg}"
        step_size_list = np.logspace(
            args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1
        )

    if alg in ["cso", "clerr"]:
        assert (
            args.cl_min is not None and args.cl_max is not None
        ), f"You did not provide --cl_min or --cl_max for algorithm {alg}"
        clip_level_list = np.logspace(
            args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1
        )

    if alg in ["nastya", "clerr"]:
        assert (
            args.in_lr_min is not None and args.in_lr_max is not None
        ), f"You did not provide --in_lr_min or --in_lr_max for algorithm {alg}!"
        in_step_size_list = np.logspace(
            args.in_lr_min, args.in_lr_max, args.in_lr_max - args.in_lr_min + 1
        )

    if alg == "clerr_heuristic":
        assert (
            args.in_lr_min is not None
            and args.in_lr_max is not None
            and args.in_lr_min == args.in_lr_max
        ), "For clerr_heuristic we fix inner learning rate, which is equal to the best learning rate of cso"
        in_step_size = [float(10**args.in_lr_min)]
        assert (
            args.in_cl is not None
        ), "You did not provide --in_cl for algorithm clerr_heuristic!"
        in_clip_level = [float(10**args.in_cl)]

        assert (
            args.lr_min is not None
            and args.lr_max is not None
            and args.cl_min is not None
            and args.cl_max is not None
        ) or (
            args.c_0_min is not None
            and args.c_0_max is not None
            and args.c_1_min is not None
            and args.c_1_max is not None
        ), f"You should either provide lr with cl or c_0 with c_1 for {alg}"

        if args.lr_min is not None and args.lr_max is not None:
            step_size_list = np.logspace(
                args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1
            )
            clip_level_list = np.logspace(
                args.cl_min, args.cl_max, args.cl_max - args.cl_min + 1
            )
            c_0_list = [None]
            c_1_list = [None]
        if args.c_0_min is not None and args.c_0_max is not None:
            c_0_list = np.logspace(
                args.c_0_min, args.c_0_max, args.c_0_max - args.c_0_min + 1
            )
            c_1_list = np.logspace(
                args.c_1_min, args.c_1_max, args.c_1_max - args.c_1_min + 1
            )
            step_size_list = [None]
            clip_level_list = [None]

    if args.cuda != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["XjA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_seeds = 3
    partial_train = partial(
        train_shuffling, args.test, task, model_type, args.add_het, alg, n_seeds, n_epochs, batch_size
    )
    if alg == "so":
        if args.n_cpus == 1:
            for lr in step_size_list:
                partial_train(lr)
        else:
            pool = Pool(args.n_cpus)
            pool.map(partial_train, step_size_list)

    elif alg == "cso":
        args_product = list(product(step_size_list, clip_level_list))
        if args.n_cpus == 1:
            for lr, cl in args_product:
                partial_train(lr, cl)
        else:
            pool = Pool(args.n_cpus)
            pool.starmap(partial_train, args_product)

    elif alg == "nastya":
        clip_level_list = [None]
        args_product = list(product(step_size_list, clip_level_list, in_step_size_list))
        if args.n_cpus == 1:
            for lr, cl, in_lr in args_product:
                partial_train(lr, cl, in_lr)
        else:
            pool = Pool(args.n_cpus)
            pool.starmap(partial_train, args_product)

    elif alg == "clerr":
        args_product = list(product(step_size_list, clip_level_list, in_step_size_list))
        if args.n_cpus == 1:
            for lr, cl, in_lr in args_product:
                partial_train(lr, cl, in_lr)
        else:
            pool = Pool(args.n_cpus)
            pool.starmap(partial_train, args_product)

    elif alg == "clerr_heuristic":
        args_product = list(
            product(
                step_size_list,
                clip_level_list,
                in_step_size,
                in_clip_level,
                c_0_list,
                c_1_list,
            )
        )
        if args.n_cpus == 1:
            for lr, cl, in_lr, in_cl, c_0, c_1 in args_product:
                partial_train(lr, cl, in_lr, in_cl, c_0, c_1)
        else:
            pool = Pool(args.n_cpus)
            pool.starmap(partial_train, args_product)
