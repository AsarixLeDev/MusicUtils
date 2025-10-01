# -*- coding: utf-8 -*-
# soundrestorer/utils/signal.py
from __future__ import annotations

from typing import Optional

import torch


def db_to_amp(db: torch.Tensor) -> torch.Tensor:
    return (10.0 ** (db / 20.0))


def amp_to_db(amp: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 20.0 * torch.log10(amp.clamp_min(eps))


def pow_to_db(power: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 10.0 * torch.log10(power.clamp_min(eps))


def add_noise_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns clean + scaled_noise; noise is scaled so that SNR(clean, scaled_noise) == snr_db
    Works with [T]/[C,T]/[B,C,T]; shapes are matched by trimming/padding noise along time.
    """
    from .audio import ensure_3d, match_length
    c3 = ensure_3d(clean)
    n3 = ensure_3d(noise)
    c3, n3 = match_length(c3, n3)
    # power per item
    B = c3.shape[0]
    c_f = c3.view(B, -1)
    n_f = n3.view(B, -1)
    Pc = (c_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    # desired Pn such that 10log10(Pc/Pn) = snr_db  =>  Pn = Pc / 10^(snr/10)
    Pn = Pc / (10.0 ** (snr_db / 10.0))
    Pn_curr = (n_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    scale = (Pn / Pn_curr).sqrt()  # per item
    scale = scale.view(B, 1, 1)
    return c3 + n3 * scale


def stft_complex(
        x: torch.Tensor, n_fft: int = 1024, hop_length: Optional[int] = None,
        win_length: Optional[int] = None, window: Optional[torch.Tensor] = None,
        center: bool = True, pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Wrapper around torch.stft that returns complex tensor shape [..., F, T].
    Accepts [T]/[C,T]/[B,C,T].
    """
    if hop_length is None: hop_length = n_fft // 4
    if win_length is None: win_length = n_fft
    x3 = x
    if x3.dim() == 1: x3 = x3.view(1, 1, -1)
    if x3.dim() == 2: x3 = x3.unsqueeze(0)
    B, C, T = x3.shape
    x2 = x3.reshape(B * C, T)
    X = torch.stft(
        x2, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, pad_mode=pad_mode, return_complex=True,
    )
    Fbins, Frames = X.shape[-2], X.shape[-1]
    return X.view(B, C, Fbins, Frames)


def istft_complex(
        X: torch.Tensor, n_fft: int = 1024, hop_length: Optional[int] = None,
        win_length: Optional[int] = None, window: Optional[torch.Tensor] = None,
        length: Optional[int] = None, center: bool = True,
) -> torch.Tensor:
    """
    Inverse STFT for complex input [..., F, T]. Returns [B, C, T].
    """
    if hop_length is None: hop_length = n_fft // 4
    if win_length is None: win_length = n_fft
    if X.dim() == 3:  # [C,F,T] -> [1,C,F,T]
        X = X.unsqueeze(0)
    if X.dim() == 2:  # [F,T] -> [1,1,F,T]
        X = X.unsqueeze(0).unsqueeze(0)
    B, C, Fbins, Frames = X.shape
    X2 = X.view(B * C, Fbins, Frames)
    y = torch.istft(
        X2, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, length=length, center=center, return_complex=False,
    )
    return y.view(B, C, -1)
