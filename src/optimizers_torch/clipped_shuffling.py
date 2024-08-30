import torch
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_

class ClippedSGD(SGD):
    def __init__(self, clip_level, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_level = clip_level

    def step(self, closure=None):
        clip_grad_norm_(self.param_groups[0]['params'], self.clip_level)
        super().step(closure)