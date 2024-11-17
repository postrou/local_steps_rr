import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from src.optimizers_torch import ShuffleOnceSampler, ClERR, NASTYA
from src.loss_functions.models import LinearModel
from .cifar import cifar_epoch_result, cifar_train_epoch


def logreg_load_data(path, batch_size):
    X, y = load_svmlight_file(path)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=0)
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    if 'covtype' in path:
        y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)
    elif 'gisette' in path or 'real-sim' in path:
        y_train_tensor = torch.tensor((y_train + 1) / 2, dtype=torch.long)
        y_test_tensor = torch.tensor((y_test + 1) / 2, dtype=torch.long)
    else:
        raise NotImplementedError(f'Check labels for dataset {path}!')

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    
    train_sampler = ShuffleOnceSampler(train_data)
    train_loader = DataLoader(train_data, batch_size, sampler=train_sampler, num_workers=10, pin_memory=True)
    train_bs = min(batch_size, len(X_train_tensor))
    if train_bs == len(X_train_tensor):
        train_loader = [next(iter(train_loader))]

    return train_loader, [(X_train_tensor, y_train_tensor)], [(X_test_tensor, y_test_tensor)]
    

def build_linear_model(input_dim, output_dim, device):
    linear_model = LinearModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    return linear_model, criterion

    
def logreg_epoch_result(*args, **kwargs):
    return cifar_epoch_result(*args, **kwargs)


def logreg_train_epoch(*args, **kwargs):
    return cifar_train_epoch(*args, **kwargs)