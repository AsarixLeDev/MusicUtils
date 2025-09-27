from typing import Dict, Any, List, Tuple
import torch
from .base import LossModule
from ..core.registry import LOSSES

class ComposedLoss:
    """
    Build from config:
      losses:
        items:
          - {name: mrstft, weight: 1.0}
          - {name: l1_wave, weight: 0.5}
          - {name: sisdr_ratio, weight: 0.35, args: {min_db: 10}}
          ...
    """
    def __init__(self, items: List[Dict[str,Any]]):
        self.items = []
        for it in items:
            name = it["name"]; w = float(it.get("weight", 1.0)); args = it.get("args", {})
            loss_fn: LossModule = LOSSES.build(name, **args)
            self.items.append((name, w, loss_fn))

    def __call__(self, outputs: Dict[str, torch.Tensor], batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        tot = 0.0; comps = {}
        for name, w, fn in self.items:
            v = fn(outputs, batch)
            if not torch.is_tensor(v): v = torch.as_tensor(v, device=outputs["yhat"].device)
            tot = tot + w * v
            comps[name] = float(v.detach().item())
        return tot, comps

    def set_attr(self, loss_name: str, attr: str, value) -> bool:
        for name, _, fn in self.items:
            if name == loss_name and hasattr(fn, attr):
                setattr(fn, attr, value)
                return True
        return False
