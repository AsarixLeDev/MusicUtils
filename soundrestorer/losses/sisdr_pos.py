# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _flatten_audio(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(1) == 1:
        x = x[:, 0]
    return x


def _si_sdr_parts(est: torch.Tensor, ref: torch.Tensor, eps: float):
    # Zero-mean
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    # Projection of est onto ref
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_pow = (ref ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / ref_pow) * ref
    e = est - s_target
    return s_target, e


class SISDRPositiveLoss(nn.Module):
    """
    Positive SI-SDR loss (error ratio):
       loss = mean( ||e||^2 / (||s_target||^2 + eps) )
    Also reports human-readable SI-SDR dB and ΔSI vs noisy if available.
    """

    def __init__(self, eps: float = 1e-8, cap: float = 1e6):
        super().__init__()
        self.eps = float(eps)
        self.cap = float(cap)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        yhat = _flatten_audio(outputs["yhat"]).to(torch.float32)
        tgt = _flatten_audio(batch["clean"]).to(torch.float32)

        s_t, e = _si_sdr_parts(yhat, tgt, self.eps)
        p_e = (e ** 2).sum(dim=-1)
        p_s = (s_t ** 2).sum(dim=-1) + self.eps
        ratio = (p_e / p_s).clamp_max(self.cap)
        loss = ratio.mean()

        # Human-readable SI-SDR dB (not used in optimization)
        si_db = 10.0 * torch.log10((p_s / (p_e + self.eps)).clamp_min(self.eps)).mean()

        # ΔSI vs noisy if batch["noisy"] exists
        delta_db = torch.tensor(0.0, device=yhat.device)
        if "noisy" in batch:
            noisy = _flatten_audio(batch["noisy"]).to(torch.float32)
            s_tn, en = _si_sdr_parts(noisy, tgt, self.eps)
            p_en = (en ** 2).sum(dim=-1)
            p_sn = (s_tn ** 2).sum(dim=-1) + self.eps
            si_noisy_db = 10.0 * torch.log10((p_sn / (p_en + self.eps)).clamp_min(self.eps)).mean()
            delta_db = si_db - si_noisy_db

        return loss, {
            "sisdr_pos": loss,
            "sisdr.db": si_db,
            "sisdr.delta_db": delta_db,
        }
