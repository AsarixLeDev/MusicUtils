# soundrestorer/ema/ema.py
import torch
from torch import nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        # keep CPU copies to avoid device/inference flag issues
        self.shadow = {
            k: v.detach().clone().cpu()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().cpu(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        # hand out plain CPU tensors
        return {k: v.detach().clone().cpu() for k, v in self.shadow.items()}
