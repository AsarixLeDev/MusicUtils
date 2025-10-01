# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple
import torch, torch.nn as nn


def _flatten_audio(x: torch.Tensor) -> torch.Tensor:
    # Accept (B,T) or (B,1,T)
    if x.dim() == 3 and x.size(1) == 1:
        x = x[:, 0]
    return x


class L1WaveLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        y = _flatten_audio(outputs["yhat"]).to(torch.float32)
        t = _flatten_audio(batch["clean"]).to(torch.float32)
        loss = (y - t).abs().mean()
        return loss, {"l1_wave": loss}
