import torch
from torch.optim import SGD


class NASTYA(SGD):
    def __init__(self, outer_lr, n_batches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_lr = outer_lr
        self.n_batches = n_batches
        self.init_g()
        
    def init_g(self):
        self.g = []
        for pg in self.param_groups:
            self.g.append([torch.zeros_like(p, requires_grad=False) for p in pg['params']])
        
    def step(self, closure=None):
        super().step(closure)
        self.update_g()

    def update_g(self):
        for i, pg in enumerate(self.param_groups):
           for j, p in enumerate(pg['params']):
                self.g[i][j] += p.grad.detach() / self.n_batches

    def outer_step(self, x_start_epoch):
        with torch.no_grad():
            for i, pg in enumerate(self.param_groups):
                for p, x_p, g_p in zip(pg['params'], x_start_epoch, self.g[i]):
                    p.grad.zero_()
                    p.copy_(x_p - self.outer_lr * g_p).requires_grad_()