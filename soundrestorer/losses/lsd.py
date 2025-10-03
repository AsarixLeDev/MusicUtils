# soundrestorer/losses/lsd.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

# shared window helper (cached per device/dtype)
from soundrestorer.utils.audio import get_stft_window


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    """
    Force audio to (B,T) float32:
      (T,)      -> (1,T)
      (B,T)     -> (B,T)
      (B,1,T)   -> (B,T)
      (B,C,T)   -> (B,T) mean over C
    """
    x = x.to(torch.float32)
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        if x.size(1) == 1:
            return x[:, 0, :]
        return x.mean(dim=1)
    raise RuntimeError(f"LSD: unexpected waveform shape {tuple(x.shape)}")


def _stft_mag_bt(x_bt: torch.Tensor,
                 n_fft: int, hop: int, win: int,
                 center: bool) -> torch.Tensor:
    """
    x_bt: (B,T) float32 -> |STFT| as (B,F,Tf) float32
    """
    win_t = get_stft_window("hann", win, device=x_bt.device, dtype=x_bt.dtype, periodic=True)
    S = torch.stft(x_bt, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=win_t, center=center, normalized=False, return_complex=True)
    return torch.abs(S)


class LogSpectralDistance(nn.Module):
    """
    Log-Spectral Distance (LSD) — positive-only, 0 at perfect reconstruction.

    loss = mean_{B,F,T}  | log10(|Y| + eps) - log10(|T| + eps) |^p
           (default p=2, i.e., MSE on log-magnitude)

    Args:
      n_fft, hop, win   : STFT params
      center            : STFT center flag
      eps               : floor to avoid log(0)
      p                 : 1 -> L1 on logmag, 2 -> MSE on logmag
    """
    def __init__(self,
                 n_fft: int = 1024,
                 hop: int = 256,
                 win: int = 1024,
                 center: bool = True,
                 eps: float = 1e-6,
                 p: int = 2):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        self.center = bool(center)
        self.eps = float(eps)
        self.p = int(p)

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y = outputs["yhat"]
        t = batch["clean"]
        y_bt = _as_bt(y)
        t_bt = _as_bt(t)

        Y = _stft_mag_bt(y_bt, self.n_fft, self.hop, self.win, self.center).clamp_min(self.eps)
        T = _stft_mag_bt(t_bt, self.n_fft, self.hop, self.win, self.center).clamp_min(self.eps)

        dlog = torch.log10(Y) - torch.log10(T)            # (B,F,Tf)
        if self.p == 1:
            d = dlog.abs()
        else:
            d = dlog.pow(2)

        per_ex = d.mean(dim=(-2, -1))                     # reduce F,T -> (B,)
        loss = per_ex.mean()                              # -> scalar

        return loss, {"lsd": loss.detach()}
