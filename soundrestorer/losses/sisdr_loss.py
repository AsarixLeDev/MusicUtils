from __future__ import annotations
import torch
from .base import LossModule
from ..core.registry import LOSSES

@LOSSES.register("sisdr")
class SISDRLoss(LossModule):
    """
    Scale-Invariant SDR loss (maximize SI-SDR):
        loss = - SI-SDR(yhat, clean)  [dB]
    Returns both the scalar loss and a comps dict {'sisdr': loss}.
    """
    def __init__(self, reduction: str = "mean"):
        self.reduction = str(reduction)

    def forward(self, outputs, batch, eps: float = 1e-8):
        y = outputs["yhat"]  # (B,T) or (B,C,T)
        x = outputs["clean"]
        if y.dim() == 3: y = y.mean(dim=1)
        if x.dim() == 3: x = x.mean(dim=1)

        # zero-mean
        y = y - y.mean(dim=-1, keepdim=True)
        x = x - x.mean(dim=-1, keepdim=True)

        # projection
        s = (torch.sum(y * x, dim=-1, keepdim=True) /
             (torch.sum(x * x, dim=-1, keepdim=True) + eps)) * x
        e = y - s
        num = torch.sum(s * s, dim=-1).clamp_min(eps)
        den = torch.sum(e * e, dim=-1).clamp_min(eps)
        sisdr_db = 10.0 * torch.log10(num / den)

        loss_vec = -sisdr_db   # maximize SI-SDR
        loss = loss_vec.mean() if self.reduction == "mean" else loss_vec
        return loss, {"sisdr": loss}
