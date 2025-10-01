# soundrestorer/losses/mrstft.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize waveform to shape [B, T] (float32), accepting:
      [T], [C,T], [B,T], [B,1,T], [B,C,T]
    For multi-channel inputs, average across channels.
    """
    x = x.to(torch.float32)
    if x.dim() == 1:                  # [T]
        return x.unsqueeze(0)         # -> [1,T]
    if x.dim() == 2:
        # Could be [B,T] or [C,T]. Heuristic: small first dim often means channels.
        if x.size(0) <= 4 and x.size(0) != x.size(1):
            return x.mean(dim=0, keepdim=True)   # -> [1,T]
        return x                                  # assume [B,T]
    if x.dim() == 3:                  # [B,C,T]
        return x.mean(dim=1)          # -> [B,T]
    raise ValueError(f"_as_bt: unsupported shape {tuple(x.shape)}")


def _stft_mag_bt(x_bt: torch.Tensor, n_fft: int, hop: int, win: int, center: bool) -> torch.Tensor:
    """
    x_bt: [B,T] float32 -> |STFT| as [B,F,Tf]
    """
    win_t = torch.hann_window(win, device=x_bt.device, dtype=x_bt.dtype)
    X = torch.stft(
        x_bt, n_fft=n_fft, hop_length=hop, win_length=win,
        window=win_t, center=center, return_complex=True
    )
    return X.abs()


class MRSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss:
      For each resolution r:
        L_r = alpha * L1(|Y|-|T|) + beta * SC(|Y|,|T|)
      MRSTFT = mean_r L_r

      SC (spectral convergence) = || |Y|-|T| ||_F / (|| |T| ||_F + eps)

    All terms are non-negative and → 0 when yhat == clean (up to numerical floors).
    """
    def __init__(self,
                 fft_sizes: List[int] = (1024, 2048, 512),
                 hops:      List[int] = (256,  512,  128),
                 win_lengths: List[int] = (1024, 2048, 512),
                 alpha: float = 0.5,
                 beta:  float = 0.5,
                 center: bool = True,
                 eps: float = 1e-8):
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
        scs  = []

        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.win_lengths):
            Y = _stft_mag_bt(y_bt, n_fft, hop, win, self.center)  # [B,F,Tf]
            T = _stft_mag_bt(t_bt, n_fft, hop, win, self.center)

            diff = (Y - T).abs()                   # [B,F,Tf]
            l_mag = diff.mean(dim=(-2, -1))       # [B]  (L1 over F,T)

            # Frobenius norms over (F,T) per batch (no 'ord' strings; works on all Torch versions)
            num = (diff.pow(2).sum(dim=(-2, -1)) + self.eps).sqrt()   # [B]
            den = (T.pow(2).sum(dim=(-2, -1)) + self.eps).sqrt()      # [B]
            sc  = num / den                                           # [B]

            Lr = self.alpha * l_mag + self.beta * sc                  # [B]
            losses.append(Lr.mean())
            mags.append(l_mag.mean())
            scs.append(sc.mean())

        loss = torch.stack(losses).mean()
        return loss, {
            "mrstft": loss.detach(),
            "mrstft.mag": torch.stack(mags).mean().detach(),
            "mrstft.sc":  torch.stack(scs).mean().detach(),
        }
