import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, learning_rate, min_lr, **kwargs):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        # 获取当前迭代次数
        it = self.last_epoch
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return [self.learning_rate * (it + 1) / (self.warmup_iters + 1) for _ in self.optimizer.param_groups]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return [self.min_lr for _ in self.optimizer.param_groups]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return [self.min_lr + coeff * (self.learning_rate - self.min_lr) for _ in self.optimizer.param_groups]
