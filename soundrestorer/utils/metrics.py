# -*- coding: utf-8 -*-
# soundrestorer/utils/metrics.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


# ---------- basic energy / loudness ----------

def rms(x: torch.Tensor, dim: Sequence[int] = (-1,), keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    y = x
    for d in sorted([d if d >= 0 else y.dim() + d for d in dim], reverse=True):
        y = y.pow(2).mean(dim=d, keepdim=True)
    y = (y + eps).sqrt()
    return y if keepdim else y.squeeze(dim)


def rms_db(x: torch.Tensor, dim: Sequence[int] = (-1,), keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    val = rms(x, dim=dim, keepdim=keepdim, eps=eps)
    out = 20.0 * torch.log10(val + eps)
    return out


# ---------- error metrics ----------

def mae(x: torch.Tensor, y: torch.Tensor, dim: Optional[Sequence[int]] = None, keepdim: bool = False) -> torch.Tensor:
    d = (x - y).abs()
    if dim is None:
        return d.mean()
    for k in sorted([dd if dd >= 0 else d.dim() + dd for dd in dim], reverse=True):
        d = d.mean(dim=k, keepdim=True)
    return d if keepdim else d.squeeze(dim)


def mse(x: torch.Tensor, y: torch.Tensor, dim: Optional[Sequence[int]] = None, keepdim: bool = False) -> torch.Tensor:
    d = (x - y) ** 2
    if dim is None:
        return d.mean()
    for k in sorted([dd if dd >= 0 else d.dim() + dd for dd in dim], reverse=True):
        d = d.mean(dim=k, keepdim=True)
    return d if keepdim else d.squeeze(dim)


# ---------- SNR ----------

def snr_db(noisy: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SNR = 10*log10( ||clean||^2 / ||noisy-clean||^2 ) computed per item.
    Accepts [T], [C,T], [B,C,T]. Returns shape [B].
    """
    nc = noisy
    cl = clean
    # unify shape
    if nc.dim() == 1: nc = nc.view(1, 1, -1)
    if cl.dim() == 1: cl = cl.view(1, 1, -1)
    if nc.dim() == 2: nc = nc.unsqueeze(0)
    if cl.dim() == 2: cl = cl.unsqueeze(0)
    B = nc.shape[0]
    # flatten over C,T
    nc_f = nc.view(B, -1)
    cl_f = cl.view(B, -1)
    err = (nc_f - cl_f)
    num = (cl_f.pow(2).sum(dim=-1) + eps)
    den = (err.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(num / den)


# ---------- SI-SDR ----------

def _flatten_bt(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if x.dim() == 1:
        return x.view(1, -1), 1
    if x.dim() == 2:
        return x.view(1, -1), 1
    if x.dim() == 3:
        B = x.shape[0]
        return x.view(B, -1), B
    raise ValueError(f"si_sdr_db: bad shape {x.shape}")


def _bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3: return x.mean(dim=1)
    raise RuntimeError(f"expected 1/2/3D audio, got {tuple(x.shape)}")


@torch.no_grad()
def si_sdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = _bt(y);
    x = _bt(x)
    xz = x - x.mean(dim=-1, keepdim=True)
    yz = y - y.mean(dim=-1, keepdim=True)
    s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
         (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
    e = yz - s
    num = torch.sum(s ** 2, dim=-1)
    den = torch.sum(e ** 2, dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)
