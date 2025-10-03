# soundrestorer/losses/composed.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn

from .l1_wave import L1WaveLoss
from .mrstft import MRSTFTLoss
from .sisdr_pos import SISDRPositiveLoss  # adjust name if your file defines a different class
from .lsd import LogSpectralDistance  # <-- add this
from .mask_unity_reg import MaskUnityReg

# in REG mapping (or equivalent), add:
REG = {
    "mrstft": MRSTFTLoss,
    "l1_wave": L1WaveLoss,
    "sisdr_pos": SISDRPositiveLoss,   # if you have it
    "lsd": LogSpectralDistance,       # <-- add this line
    "mask_unity_reg": MaskUnityReg, # if present
}



class ComposedLoss(nn.Module):
    def __init__(self, items: List[Dict[str, Any]]):
        super().__init__()
        self.entries = nn.ModuleList()
        self.weights = []
        for it in items:
            name = it["name"]
            cls = REG[name]
            self.entries.append(cls(**(it.get("args") or {})))
            self.weights.append(float(it.get("weight", 1.0)))

    def forward(self, outputs, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.zeros((), device=outputs["yhat"].device, dtype=outputs["yhat"].dtype)
        comps = {}
        for w, loss in zip(self.weights, self.entries):
            val = loss(outputs, batch)
            if isinstance(val, tuple):
                v, _ = val
            else:
                v = val
            total = total + w * v
        comps["total"] = float(total.detach().cpu())
        return total, comps


def build_losses(cfg: Dict[str, Any]) -> nn.Module:
    items = (cfg.get("losses") or {}).get("items", [])
    return ComposedLoss(items)
