import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp

class GradientAccumulator:
    def __init__(self, steps=1, enabled=True):
        self.steps = steps
        self.enable = enabled
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def inc_counter(self):
        self._counter += 1
        self._counter %= self.steps

    @property
    def is_start_cycle(self):
        return self._counter == 0

    @property
    def is_end_cycle(self):
        return self._counter == self.steps - 1

    def backward_step(self, model: nn.Module, loss: torch.Tensor,
                      optimizer: optim.Optimizer, scaler: amp.GradScaler):
        if not self.enable:
            return
        if optimizer is None:
            return

        loss = loss / self.steps

        if self.is_start_cycle:
            # if pytorch version >= 1.7, set set_to_none=True for better performance
            optimizer.zero_grad(set_to_none=True)

        if isinstance(model, nn.parallel.DistributedDataParallel) and not self.is_end_cycle:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        if self.is_end_cycle:
            scaler.step(optimizer)
            scaler.update()

        self.inc_counter()