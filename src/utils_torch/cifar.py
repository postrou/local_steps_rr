import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from src.optimizers_torch import ShuffleOnceSampler, ClERR, NASTYA
from src.loss_functions.models import ResNet18, LeNet5


def cifar_load_data(path, batch_size, add_het=False):
    print("Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_data = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True
    )
    if add_het:
        cifar_add_heterogeneity(train_data)
    train_data.transform = transform_train
    train_sampler = ShuffleOnceSampler(train_data)
    train_bs = min(batch_size, len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_bs, sampler=train_sampler, num_workers=1, pin_memory=True
    )
    if train_bs == len(train_data):
        train_loader = [next(iter(train_loader))]

    train_ep_eval_loader = torch.utils.data.DataLoader(
        train_data, batch_size=len(train_data), sampler=train_sampler, num_workers=10, pin_memory=True
    )
    train_data = [next(iter(train_ep_eval_loader))]

    test_data = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    test_sampler = ShuffleOnceSampler(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), sampler=test_sampler, num_workers=10, pin_memory=True
    )
    test_data = [next(iter(test_loader))]

    return train_loader, train_data, test_data


def cifar_add_heterogeneity(train_data):
    print('Adding heterogeneity...')
    train_data_per_class = {i: [] for i in range(10)}
    for i, (im, cl) in enumerate(train_data):
        train_data_per_class[cl].append(i)

    val = 127
    for cl in train_data_per_class:
        cl_pic_idx = train_data_per_class[cl]
        chosen_pic_idx = np.random.choice(cl_pic_idx, int(0.5 * len(cl_pic_idx)), replace=False)
        for pic_id in chosen_pic_idx:
            is_row = np.random.choice([True, False])
            is_add = np.random.choice([True, False])
            pic = train_data.data[pic_id]
            if is_row:
                row_idx = np.random.choice(range(pic.shape[1]), int(0.4 * pic.shape[1]), replace=False)
                if is_add:
                    pic[row_idx, :, :] = np.minimum(pic[row_idx, :, :] + val, 255)
                else:
                    pic[row_idx, :, :] = np.maximum(pic[row_idx, :, :] - val, 0)
            else:
                col_idx = np.random.choice(range(pic.shape[2]), int(0.4 * pic.shape[2]), replace=False)
                if is_add:
                    pic[:, col_idx, :] = np.minimum(pic[:, col_idx, :] + val, 255)
                else:
                    pic[:, col_idx, :] = np.maximum(pic[:, col_idx, :] - val, 0)
        

def build_resnet_model(device):
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def build_lenet_model(device):
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def cifar_predict_and_loss(inputs, targets, model, criterion):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        loss = criterion(outputs, targets)
    return predicted, loss


def cifar_epoch_result(train_data, test_data, model, criterion, device):
    model.eval()
    model.zero_grad()
    train_loss = 0
    train_acc = 0
    for inputs, targets in train_data:
        inputs, targets = inputs.to(device), targets.to(device)
        # here we get mean loss over the batch
        predicted, loss = cifar_predict_and_loss(inputs, targets, model, criterion)
        loss.backward()
        train_loss += loss.item()
        train_acc += predicted.eq(targets).sum().item()
    train_acc /= len(inputs)
    train_grad_norm = model.gradient_norm()

    test_loss = 0
    test_acc = 0
    model.zero_grad()
    for inputs, targets in test_data:
        inputs, targets = inputs.to(device), targets.to(device)
        # here we get mean loss over the batch
        predicted, loss = cifar_predict_and_loss(inputs, targets, model, criterion)
        loss.backward()
        test_loss += loss.item()
        test_acc += predicted.eq(targets).sum().item()
    test_acc /= len(inputs)
    test_grad_norm = model.gradient_norm()
    return train_loss, train_acc, train_grad_norm, test_loss, test_acc, test_grad_norm


def cifar_train_epoch(progress_bar, model, criterion, optimizer, train_local_loss_vals,
                train_local_gns, train_local_acc_vals, initial_loss,
                initial_gn, initial_acc, device):
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        predicted, loss = cifar_predict_and_loss(inputs, targets, model, criterion)
        optimizer.zero_grad()
        loss.backward()

        local_loss = loss.item()
        local_gn = model.gradient_norm()
        local_acc = predicted.eq(targets).sum().item() / len(targets)
        if initial_loss is None or initial_gn is None or initial_acc is None:
            initial_loss = local_gn
            initial_gn = local_gn
            initial_acc = local_acc

        optimizer.step()
        if type(optimizer) in [ClERR, NASTYA]:
            optimizer.update_g()

        progress_bar.set_postfix(
            l_loss=f"{initial_loss:.3f}->{local_loss:.3f}",
            l_gn=f"{initial_gn:.3f}->{local_gn:.3f}",
            l_acc=f"{initial_acc:.3f}->{local_acc:.3f}",
        )
        train_local_loss_vals.append(local_loss)
        train_local_gns.append(local_gn)
        train_local_acc_vals.append(local_acc)



