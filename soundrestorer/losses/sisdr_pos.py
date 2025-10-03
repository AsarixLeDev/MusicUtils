# soundrestorer/losses/sisdr_pos.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn

# use your robust SI-SDR (db) helper
from soundrestorer.utils.metrics import si_sdr_db


class SISDRPositiveLoss(nn.Module):
    """
    Positive-only SI-SDR penalty (0 when target met or exceeded).

    Formula (per sample):
      si = SI-SDR_dB(yhat, clean)
      loss_i = relu((target_db - si) / (norm_db))         # norm_db defaults to target_db
    Then mean over batch.

    Args
    ----
    target_db : float
        Desired SI-SDR in dB. If prediction achieves >= target_db, penalty is 0.
    norm_db : float
        Normalization scale for the penalty. Defaults to target_db.
    eps : float
        Numerical floor.
    **_ : any
        Extra/unknown kwargs are ignored for config compatibility.
    """
    def __init__(self, target_db: float = 20.0, norm_db: float | None = None, eps: float = 1e-8, **_):
        super().__init__()
        self.target_db = float(target_db)
        self.norm_db = float(norm_db) if norm_db is not None else float(target_db)
        self.eps = float(eps)

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        yhat = outputs["yhat"]
        clean = batch["clean"]
        # compute per-sample SI-SDR in dB; ensure mono/batch handled inside helper
        si = si_sdr_db(yhat, clean, eps=self.eps)    # (B,) or scalar tensor
        # positive-only penalty
        penalty = torch.clamp((self.target_db - si) / self.norm_db, min=0.0)
        loss = penalty.mean()
        return loss, {"sisdr_pos": loss.detach()}
