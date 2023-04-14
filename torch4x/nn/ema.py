import copy

import torch
import torch.nn as nn


class ModelEma(nn.Module):
    """ 
    Copy from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py#L82
    License under Apache License 2.0, full license text can be found at 
    https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
    """

    def __init__(self, model: nn.Module, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model: nn.Module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model: nn.Module):
        self._update(model, update_fn=lambda e, m: m)
