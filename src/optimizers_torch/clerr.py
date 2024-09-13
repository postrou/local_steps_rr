import torch
from torch.nn.utils import clip_grad_norm_

from .nastya import NASTYA
from .clipped_shuffling import ClippedSGD


class ClERR(NASTYA):
    def __init__(self, c_0, c_1, n_batches, use_g_in_outer_step=False, *args, **kwargs):
        super(NASTYA, self).__init__(*args, **kwargs)
        self.c_0 = c_0
        self.c_1 = c_1
        self.use_g_in_outer_step = use_g_in_outer_step
        self.n_batches = n_batches
        self.init_g()

    def calculate_outer_lr(self, norm_grad_start_epoch):
        if self.use_g_in_outer_step:
            assert len(self.g) == 1, \
                'Current version of outer step calculation is implemented only for single param group!'
            outer_step_size = \
                1 / (self.c_0 + self.c_1 * self.grad_norm(self.g[0]))
        else:
            assert norm_grad_start_epoch is not None
            outer_step_size = \
                1 / (self.c_0 + self.c_1 * norm_grad_start_epoch)
        return outer_step_size

    def outer_step(self, x_start_epoch, norm_grad_start_epoch=None):
        self.outer_lr = self.calculate_outer_lr(norm_grad_start_epoch)
        with torch.no_grad():
            for i, pg in enumerate(self.param_groups):
                for p, x_p, g_p in zip(pg['params'], x_start_epoch, self.g[i]):
                    p.grad.zero_()
                    p.copy_(x_p - self.outer_lr * g_p).requires_grad_()
    
    def grad_norm(self, param_grads):
        gn = 0
        for pg in param_grads:
            gn += pg.square().sum()
        return gn.sqrt()


class ClERRHeuristic(ClERR):
    def __init__(self, c_0, c_1, in_clip_level, *args, **kwargs):
        super().__init__(c_0, c_1, None, True, *args, **kwargs)
        self.in_clip_level = in_clip_level

    def step(self, closure=None):
        gn = self.grad_norm(self.param_groups[0]['params'])
        clip_coef = max(1, self.in_clip_level / gn)
        clip_grad_norm_(self.param_groups[0]['params'], self.in_clip_level)
        super(NASTYA, self).step(closure)
        self.update_g(clip_coef)

    def update_g(self, clip_coef):
        for i, pg in enumerate(self.param_groups):
            gamma = clip_coef * pg['lr']
            for j, p in enumerate(pg['params']):
                self.g[i][j] += gamma * p.grad.detach()
