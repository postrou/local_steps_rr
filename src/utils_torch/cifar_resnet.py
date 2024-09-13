import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from src.optimizers_torch import ShuffleOnceSampler, ClERR, NASTYA
from src.loss_functions.models import ResNet18


def cifar_load_data(path, batch_size):
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
        root=path, train=True, download=True, transform=transform_train
    )
    train_sampler = ShuffleOnceSampler(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True
    )

    test_data = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    test_sampler = ShuffleOnceSampler(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


def build_resnet_model(device):
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def cifar_predict_and_loss(inputs, targets, model, criterion):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        loss = criterion(outputs, targets)
    return predicted, loss

def cifar_epoch_result(train_loader, test_loader, model, criterion, device):
    model.eval()
    model.zero_grad()
    train_loss = 0
    train_acc = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # here we get mean loss over the batch
        predicted, loss = cifar_predict_and_loss(inputs, targets, model, criterion)
        loss_fix_normalization = loss * len(targets) / len(train_loader.dataset)
        # this allows us to get the correct norm of the full gradient
        loss_fix_normalization.backward()
        train_loss += loss_fix_normalization.item()
        train_acc += predicted.eq(targets).sum().item()
    train_acc /= len(train_loader.dataset)
    train_grad_norm = model.gradient_norm()

    test_loss = 0
    test_acc = 0
    model.zero_grad()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # here we get mean loss over the batch
        predicted, loss = cifar_predict_and_loss(inputs, targets, model, criterion)
        loss_fix_normalization = loss * len(targets) / len(test_loader.dataset)
        # this allows us to get the correct norm of the full gradient
        loss_fix_normalization.backward()
        test_loss += loss_fix_normalization.item()
        test_acc += predicted.eq(targets).sum().item()
    test_acc /= len(test_loader.dataset)
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



