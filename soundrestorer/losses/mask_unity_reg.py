from __future__ import annotations

import torch.nn as nn
from ..core.registry import LOSSES




@LOSSES.register("mask_unity_reg")
class MaskUnityReg(nn.Module):
    """
    Penalize deviation of the *applied* mask from unity (R≈1, I≈0).
    Reads R/I from Task outputs so it matches the actual clamp/limit logic.
    Keep the weight small and decay it away in a curriculum.
    """

    def __init__(self, p: float = 2.0, **_):
        super().__init__()
        self.p = float(p)

    def forward(self, outputs, batch):
        R = outputs.get("R", None)
        I = outputs.get("I", None)
        if R is None or I is None:
            raise RuntimeError("mask_unity_reg: outputs must contain 'R' and 'I'")
        dR = (R - 1.0).abs().pow(self.p)
        dI = I.abs().pow(self.p)
        return dR.mean() + dI.mean()
