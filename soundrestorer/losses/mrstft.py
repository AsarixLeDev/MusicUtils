# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch, torch.nn as nn
import math

from ..metrics.common import _stft_mag


def _flatten_audio(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(1) == 1:
        x = x[:, 0]
    return x


class MRSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss:
      L = mean_r ( alpha * | |S_y|-|S_t| |_1 + beta * SC )
    where SC = || |S_y|-|S_t| ||_F / (|| |S_t| ||_F + eps)
    Default alpha=0.5, beta=0.5; you can tweak in YAML.
    """
    def __init__(self,
                 fft_sizes: List[int] = (1024, 2048, 512),
                 hops: List[int] = (256, 512, 128),
                 win_lengths: List[int] = (1024, 2048, 512),
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 center: bool = True):
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(win_lengths), "MRSTFT: lists must be same length."
        self.fft_sizes  = list(map(int, fft_sizes))
        self.hops       = list(map(int, hops))
        self.win_lengths= list(map(int, win_lengths))
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.center = bool(center)
        self.eps = 1e-8

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        y = _flatten_audio(outputs["yhat"]).to(torch.float32)
        t = _flatten_audio(batch["clean"]).to(torch.float32)

        losses = []
        scs = []
        mags = []

        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.win_lengths):
            Y = _stft_mag(y, n_fft, hop, win, center=self.center)
            T = _stft_mag(t, n_fft, hop, win, center=self.center)
            diff = (Y - T).abs()
            l_mag = diff.mean()

            # spectral convergence
            num = torch.linalg.vector_norm(diff, ord='fro', dim=(-2, -1))
            den = torch.linalg.vector_norm(T,    ord='fro', dim=(-2, -1))
            sc = (num / (den + self.eps)).mean()

            losses.append(self.alpha * l_mag + self.beta * sc)
            scs.append(sc)
            mags.append(l_mag)

        loss = torch.stack(losses).mean()
        return loss, {
            "mrstft": loss,
            "mrstft.mag": torch.stack(mags).mean(),
            "mrstft.sc": torch.stack(scs).mean(),
        }
