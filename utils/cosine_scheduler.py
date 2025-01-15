import math 
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, config, last_epoch=-1):
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps
        self.max_lr = config.max_lr 
        self.min_lr = config.min_lr
        self.steps = last_epoch + 1 
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.steps < self.warmup_steps:
            lr = self.max_lr * (self.steps + 1) / self.warmup_steps
        elif self.steps > self.max_steps:
            lr = self.min_lr
        else: 
            decay_ratio = (self.steps - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        self.steps += 1
        return [lr for _ in self.optimizer.param_groups]
  