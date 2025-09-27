from __future__ import annotations

from typing import Dict

import torch

__all__ = ["EMA"]


class EMA:
    """
    Device/dtype aware Exponential Moving Average of model weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        p0 = next(model.parameters())
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().to(device=p0.device, dtype=p0.dtype)
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if not torch.is_tensor(v) or not v.dtype.is_floating_point:
                continue
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                # keep shadow on same device/dtype as current param
                if self.shadow[k].device != v.device or self.shadow[k].dtype != v.dtype:
                    self.shadow[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)
                self.shadow[k].lerp_(v.detach(), 1.0 - d)

    @torch.no_grad()
    def load_to(self, model: torch.nn.Module, strict: bool = False):
        model.load_state_dict(self.shadow, strict=strict)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().cpu() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, state: Dict[str, torch.Tensor], model: torch.nn.Module | None = None):
        # Optionally map to model's param device/dtype if provided
        if model is not None:
            for k, v in model.state_dict().items():
                if k in state and torch.is_tensor(state[k]):
                    self.shadow[k] = state[k].detach().to(device=v.device, dtype=v.dtype)
                else:
                    self.shadow[k] = v.detach().clone().to(device=v.device, dtype=v.dtype)
        else:
            self.shadow = {k: v.detach().clone() for k, v in state.items()}
