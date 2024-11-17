import os
import time
import hashlib
import copy

import torch
from torch.utils.data import Dataset, DataLoader

from .penn_data import Corpus
from .splitcross import SplitCrossEntropyLoss
from src.loss_functions.models import LSTMModel
from src.optimizers_torch import RandomReshufflingSampler, ClERR, NASTYA


class PTBDataset(Dataset):
    def __init__(self, data, seq_length):
        super().__init__()
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq_length = min(self.seq_length, len(self.data) - 1 - idx)
        input_seq = self.data[idx:idx + seq_length]
        target_seq = self.data[idx + 1 : idx + seq_length + 1]
        if type(input_seq) != torch.Tensor and type(target_seq) != torch.Tensor:
            input_seq = torch.tensor(input_seq)
            target_seq = torch.tensor(target_seq)
        return input_seq, target_seq


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def penn_load_data(path, batch_size):
    fn = 'corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = Corpus(path)
        torch.save(corpus, fn)

    n_tokens = len(corpus.dictionary)
    seq_len = 70
    # eval_batch_size = 10
    # test_batch_size = 1
    train_data = PTBDataset(corpus.train, seq_len)
    test_data = PTBDataset(corpus.test, seq_len)
    train_sampler = RandomReshufflingSampler(train_data)
    test_sampler = RandomReshufflingSampler(test_data)
    train_loader = DataLoader(train_data, batch_size, sampler=train_sampler, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size, sampler=test_sampler, 
                             num_workers=4, pin_memory=True)
    return n_tokens, train_loader, test_loader


def build_lstm_model(n_tokens, device):
    emsize = 200
    nhid = 512
    nlayers = 2
    dropout = 0.4
    dropouth = 0
    dropouti = 0
    dropoute = 0
    wdrop = 0
    tied = True 
    model = LSTMModel(n_tokens, emsize, nhid, nlayers, dropout, dropouth, 
                      dropouti, dropoute, wdrop, tied)

    splits = []
    if n_tokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif n_tokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)

    model, criterion = model.to(device), criterion.to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)
    return model, criterion


def penn_epoch_result(train_loader, test_loader, model, criterion, device):
    model.eval()
    model.zero_grad()
    train_loss = 0
    print(train_loader.batch_size)
    hidden = model.init_hidden(train_loader.batch_size)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output, hidden = model(inputs, hidden)
        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        loss_fix_normalization = loss * len(targets) / len(train_loader.dataset)
        loss_fix_normalization.backward()
        train_loss += loss_fix_normalization.item()
        hidden = repackage_hidden(hidden)
    train_grad_norm = model.gradient_norm()

    test_loss = 0
    model.zero_grad()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output, hidden = model(inputs, hidden)
        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        loss_fix_normalization = loss * len(targets) / len(test_loader.dataset)
        loss_fix_normalization.backward()
        test_loss += loss_fix_normalization.item()
        hidden = repackage_hidden(hidden)
    test_grad_norm = model.gradient_norm()
    return train_loss, train_grad_norm, test_loss, test_grad_norm


def penn_train_epoch(progress_bar, model, criterion, optimizer, batch_size,
                     train_local_loss_vals,train_local_gns, initial_loss,
                     initial_gn, device):
    # Turn on training mode which enables dropout.
    alpha = 2
    beta = 1

    total_loss = 0
    hidden = model.init_hidden(batch_size)
    batch, i = 0, 0
    model.train()
    # while i < train_data.size(0):
    for data, targets in progress_bar:
        data, targets = data.to(device), targets.to(device)
        print(data.shape, targets.shape)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        outputs, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        local_loss = criterion(model.decoder.weight, model.decoder.bias, outputs, targets)

        loss = local_loss
        # Activiation Regularization
        loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()
        optimizer.step()
        if type(optimizer) in [ClERR, NASTYA]:
            optimizer.update_g()

        total_loss += local_loss.data
        
        local_gn = model.gradient_norm()
        progress_bar.set_postfix(
            l_loss=f"{initial_loss:.3f}->{local_loss:.3f}",
            l_gn=f"{initial_gn:.3f}->{local_gn:.3f}",
        )
        train_local_loss_vals.append(local_loss)
        train_local_gns.append(local_gn)

        # if batch % args.smooth_log_interval == 0 and batch > 0:
            # iterationlogger.write_row(eval_smooth(prev_model, model))
        
        # if batch % args.log_interval == 0 and batch > 0:
        #     cur_loss = total_loss.item() / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
        #             'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
        #         epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
        #         elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
        #     total_loss = 0
        #     start_time = time.time()
        ###
        # batch += 1
        # i += seq_len

