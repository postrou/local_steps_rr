import os
import argparse
import random
from multiprocessing import Pool
from functools import partial
import pickle
import sys

import torch
import numpy as np
from tqdm.auto import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from src.optimizers import ShuffleOnceSampler
from src.loss_functions.models import ResNet18


def redirect_output(log_file):
    sys.stdout = open(log_file, 'w')
    sys.stderr = open(log_file, 'w')

def load_data(path, batch_size):
    print('Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    train_sampler = ShuffleOnceSampler(train_data, batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler) 

    test_data = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test)
    test_sampler = ShuffleOnceSampler(test_data, batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader


def build_model(device):
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    
def predict_and_loss(inputs, targets, model, criterion):
    outputs = model(inputs)
    _, predicted = outputs.max(1)
    loss = criterion(outputs, targets)
    return predicted, loss


def gradient_norm(model):
    gn = 0
    for p in model.parameters():
        gn += p.grad.square().sum()
    return gn.sqrt().item()

    
def compute_and_store_epoch_results(
    epoch,
    train_loss_vals, 
    train_gns, 
    train_acc_vals,
    test_loss_vals, 
    test_gns, 
    test_acc_vals, 
    train_loader, 
    test_loader, 
    model, 
    criterion, 
    device
):
    train_loss, train_acc, train_gn, test_loss, test_acc, test_gn = \
        epoch_result(train_loader, test_loader, model, criterion, device)
    train_loss_vals[epoch + 1] = train_loss
    train_acc_vals[epoch + 1] = train_acc
    train_gns[epoch + 1] = train_gn
    test_loss_vals[epoch + 1] = test_loss
    test_acc_vals[epoch + 1] = test_acc
    test_gns[epoch + 1] = test_gn


def epoch_result(train_loader, test_loader, model, criterion, device):
    model.eval()
    model.zero_grad()
    train_loss = 0
    train_acc = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # here we get mean loss over the batch
        predicted, loss = predict_and_loss(inputs, targets, model, criterion)
        loss_fix_normalization = loss * len(targets) / len(train_loader.dataset)
        # this allows us to get the correct norm of the full gradient
        loss_fix_normalization.backward()
        train_loss += loss_fix_normalization.item()
        train_acc += predicted.eq(targets).sum().item()
    train_acc /= len(train_loader.dataset)
    train_grad_norm = gradient_norm(model)

    test_loss = 0
    test_acc = 0
    model.zero_grad()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # here we get mean loss over the batch
        predicted, loss = predict_and_loss(inputs, targets, model, criterion)
        loss_fix_normalization = loss * len(targets) / len(test_loader.dataset)
        # this allows us to get the correct norm of the full gradient
        loss_fix_normalization.backward()
        test_loss += loss_fix_normalization.item()
        test_acc += predicted.eq(targets).sum().item()
    test_acc /= len(test_loader.dataset)
    test_grad_norm = gradient_norm(model)
    return train_loss, train_acc, train_grad_norm, test_loss, test_acc, test_grad_norm


def init_seed(seed, n_epochs):
    set_seed(seed)
    train_loss_vals = [0.] * (n_epochs + 1)
    train_gns = [0.] * (n_epochs + 1)
    train_acc_vals = [0.] * (n_epochs + 1)
    train_local_loss_vals = []
    train_local_gns = []
    train_local_acc_vals = []
    test_loss_vals = [0.] * (n_epochs + 1)
    test_gns = [0.] * (n_epochs + 1)
    test_acc_vals = [0.] * (n_epochs + 1)
    return train_loss_vals, train_gns, train_acc_vals, \
        train_local_loss_vals, train_local_gns, train_local_acc_vals,\
            test_loss_vals, test_gns, test_acc_vals
 
 
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
    
    
def train_shuffling(alg, n_seeds, n_epochs, batch_size, lr):
    print(f'Starting {alg} for lr={lr}')
    log_dir = f'logs/cifar10/bs_{batch_size}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if alg == 'so':
        log_fn = f'so_lr_{lr}_seeds_{n_seeds}_{n_epochs}.log'
    log_path = os.path.join(log_dir, log_fn)
    redirect_output(log_path)

    results_dir = f'results/cifar10/bs_{batch_size}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_fn = f'so_lr_{lr}_seeds_{n_seeds}_{n_epochs}'
    results_path = os.path.join(results_dir, results_fn)
    if os.path.exists(results_path):
        print(f'Results for so, run for {n_seeds} seeds, {n_epochs} epochs with lr={lr} already exist!')
        return

    device = 'cuda'
    train_loader, test_loader = load_data('datasets/cifar10/', batch_size)
    model, criterion = build_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loss_vals_all = {}
    train_acc_vals_all = {}
    train_gns_all = {}
    train_local_loss_vals_all = {}
    train_local_acc_vals_all = {}
    train_local_gns_all = {}
    test_loss_vals_all = {}
    test_acc_vals_all = {}
    test_gns_all = {}

    # checkpoint_dir = 'checkpoints/resnet_cifar/'
    # if not os.path.exists(checkpoint_dir):
        # os.makedirs(checkpoint_dir)
   
    model.train()
    for seed in range(n_seeds):
        train_loss_vals, train_gns, train_acc_vals, \
            train_local_loss_vals, train_local_gns, train_local_acc_vals, \
                test_loss_vals, test_gns, test_acc_vals = init_seed(seed, n_epochs)
        # print('Computing and storing initial results...')
        compute_and_store_epoch_results(
            -1,
            train_loss_vals, 
            train_gns, 
            train_acc_vals,
            test_loss_vals, 
            test_gns, 
            test_acc_vals, 
            train_loader, 
            test_loader, 
            model, 
            criterion, 
            device
        )
        print(f'💪💪💪 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} | Epoch {0}/{n_epochs} | Train loss={train_loss_vals[0]:.4f} | Train gn={train_gns[0]:.4f} | Train acc={train_acc_vals[0]:.4f}')
        print(f'🧪🧪🧪 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} | Epoch {0}/{n_epochs} | Test loss={test_loss_vals[0]:.4f} | Test gn={test_gns[0]:.4f} | Test acc={test_acc_vals[0]:.4f}')

        initial_loss, initial_gn, initial_acc = None, None, None
        for epoch in range(n_epochs):
            # checkpoint_fn = f'so_lr_{lr}_seed_{seed}_epoch_{epoch}.pth'
            # checkpoint_path = os.path.join(checkpoint_dir, checkpoint_fn)
            # if not os.path.exists(checkpoint_path):
            progress_bar = tqdm(train_loader, desc=f'🤖🤖🤖 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs}', leave=True)
            for i, (inputs, targets) in enumerate(progress_bar):
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                predicted, loss = predict_and_loss(inputs, targets, model, criterion)
                loss.backward()

                local_loss = loss.item()
                local_gn = gradient_norm(model)
                local_acc = predicted.eq(targets).sum().item() / len(targets)
                if initial_loss is None or initial_gn is None or initial_acc is None:
                    initial_loss = local_gn
                    initial_gn = local_gn
                    initial_acc = local_acc

                optimizer.step()

                progress_bar.set_postfix(
                    l_loss=f'{initial_loss:.3f}->{local_loss:.3f}', 
                    l_gn=f'{initial_gn:.3f}->{local_gn:.3f}',
                    l_acc=f'{initial_acc:.3f}->{local_acc:.3f}'
                )
                train_local_loss_vals.append(local_loss)
                train_local_gns.append(local_gn)
                train_local_acc_vals.append(local_acc)

                # print(f'🤖🤖🤖 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs} finished!')
            # else:
                # print(f'Model checkpoint for SO, lr={lr}, seed #{seed + 1} and epoch #{epoch} exists, loading...')
                # model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

            # print('Computing and storing epoch results...')
            compute_and_store_epoch_results(
                epoch,
                train_loss_vals, 
                train_gns, 
                train_acc_vals,
                test_loss_vals, 
                test_gns, 
                test_acc_vals, 
                train_loader, 
                test_loader, 
                model, 
                criterion, 
                device
            )
            print(f'💪💪💪 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs} | Train loss={train_loss_vals[epoch + 1]:.4f} | Train gn={train_gns[epoch + 1]:.4f} | Train acc={train_acc_vals[epoch + 1]:.4f}')
            print(f'🧪🧪🧪 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} | Epoch {epoch + 1}/{n_epochs} | Test loss={test_loss_vals[epoch + 1]:.4f} | Test gn={test_gns[epoch + 1]:.4f} | Test acc={test_acc_vals[epoch + 1]:.4f}')
            
            # torch.save(model.state_dict(), checkpoint_path)


        print(f'🌱🌱🌱 SO | Lr={lr} | Seed {seed + 1}/{n_seeds} finished!')
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
        'train_loss': train_loss_vals_all,
        'train_gn': train_gns_all,
        'train_acc': train_acc_vals_all,
        'train_local_loss': train_local_loss_vals_all,
        'train_local_gn': train_local_gns_all,
        'train_local_acc': train_local_acc_vals_all,
        'test_loss': test_loss_vals_all,
        'test_gn': test_gns_all,
        'test_acc': test_acc_vals_all
    }

    with open(results_path, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', type=str)
    parser.add_argument('alg', type=str)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--x_opt', action='store_true')
    parser.add_argument('--cl_min', type=int, default=None, help='min clip level in log scale')
    parser.add_argument('--cl_max', type=int, default=None, help='max clip level in log scale')
    parser.add_argument('--lr_min', type=int, default=None, help='min step size in log scale')
    parser.add_argument('--lr_max', type=int, default=None, help='max step size in log scale')
    parser.add_argument('--in_lr_min', type=int, default=None, help='min inner step size in log scale (for CLERR)')
    parser.add_argument('--in_lr_max', type=int, default=None, help='max inner step size in log scale (for CLERR)')
    parser.add_argument('--use_g', action='store_true')
    parser.add_argument('--n_cpus', type=int, default=1, help='number of processes to run in parallel')
    parser.add_argument('--cuda', type=int, default=-1, help='number of cuda to use (default -1 is for cpu)')
    args = parser.parse_args()

    alg = args.alg
    assert alg == 'so', 'Other algorithms are not implemented yet!'

    assert args.lr_min is not None and args.lr_max is not None, \
        f'You did not provide --lr_min or --lr_max for algorithm {alg}'

    if args.cuda != -1:
        os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
        os.environ["XjA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    step_size_list = np.logspace(args.lr_min, args.lr_max, args.lr_max - args.lr_min + 1)
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_seeds = 3
    partial_so = partial(
        train_shuffling,
        alg,
        n_seeds,
        n_epochs,
        batch_size
    )
    if args.n_cpus == 1:
        for lr in step_size_list:
            partial_so(lr)
    else:
        pool = Pool(args.n_cpus)
        pool.map(partial_so, step_size_list)

    