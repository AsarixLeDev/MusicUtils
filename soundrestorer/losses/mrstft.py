# soundrestorer/losses/mrstft.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..metrics.common import _stft_mag, _as_bt  # keep using the centralized helper


class MRSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=(1024, 512, 2048),
                 hops=(256, 128, 512),
                 win_lengths=(1024, 512, 2048),
                 alpha: float = 0.5,
                 beta: float  = 0.5,
                 center: bool = True,
                 eps: float   = 1e-8,
                 **_):   # <-- swallow mag_loss/logmag_loss or any extra keys
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(win_lengths), "MRSTFT: list lengths must match."
        self.fft_sizes   = [int(v) for v in fft_sizes]
        self.hops        = [int(v) for v in hops]
        self.win_lengths = [int(v) for v in win_lengths]
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.center = bool(center)
        self.eps = float(eps)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        # Normalize shapes to [B,T]
        y_bt = _as_bt(outputs["yhat"])
        t_bt = _as_bt(batch["clean"])

        losses = []
        mags = []
        scs = []

        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.win_lengths):
            Yh = _stft_mag(y_bt, n_fft, hop, win=None, center=self.center)
            Y = _stft_mag(t_bt, n_fft, hop, win=None, center=self.center)

            diff = (Yh - Y).abs()  # [B,F,Tf]
            l_mag = diff.mean(dim=(-2, -1))  # [B]  (L1 over F,T)

            # Frobenius norms over (F,T) per batch (no 'ord' strings; works on all Torch versions)
            num = (diff.pow(2).sum(dim=(-2, -1)) + self.eps).sqrt()  # [B]
            den = (Y.pow(2).sum(dim=(-2, -1)) + self.eps).sqrt()  # [B]
            sc = num / den  # [B]

            Lr = self.alpha * l_mag + self.beta * sc  # [B]
            losses.append(Lr.mean())
            mags.append(l_mag.mean())
            scs.append(sc.mean())

        loss = torch.stack(losses).mean()
        return loss, {
            "mrstft": loss.detach(),
            "mrstft.mag": torch.stack(mags).mean().detach(),
            "mrstft.sc": torch.stack(scs).mean().detach(),
        }
