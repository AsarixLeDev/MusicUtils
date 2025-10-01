# -*- coding: utf-8 -*-
# soundrestorer/utils/audio.py
from __future__ import annotations
from typing import List, Sequence, Tuple, Optional
import torch

def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure a waveform tensor is [B, C, T].
    Accepts [T], [C, T], [B, C, T]. Returns a view/clone on same device/dtype.
    """
    if x.dim() == 1:
        return x.view(1, 1, -1)
    if x.dim() == 2:
        # [C, T] -> [1, C, T]
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"ensure_3d: expected 1/2/3 dims, got {x.shape}")

def to_mono(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    Mix down channels by mean across C. Works with [T], [C, T], [B, C, T].
    Returns [B, 1, T] if keepdim else [B, T].
    """
    x3 = ensure_3d(x)
    m = x3.mean(dim=1, keepdim=True)  # [B, 1, T]
    if keepdim:
        return m
    return m.squeeze(1)

def peak(x: torch.Tensor, dim: Sequence[int] = (-1,), keepdim: bool = False) -> torch.Tensor:
    xabs = x.abs()
    for d in sorted([d if d >= 0 else xabs.dim() + d for d in dim], reverse=True):
        xabs, _ = xabs.max(dim=d, keepdim=True)
    return xabs if keepdim else xabs.squeeze(dim)

def normalize_peak(x: torch.Tensor, target: float = 0.98, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale per-sample to a target peak in [-target, target].
    Works for [T], [C,T], [B,C,T] (normalized independently per item in batch).
    """
    x3 = ensure_3d(x)
    p = peak(x3, dim=(-1,), keepdim=True)  # [B, C, 1] peak per chan
    s = (target / (p + eps))
    s = torch.clamp(s, max=1e4)
    y = x3 * s
    return y if x.dim() == 3 else y.squeeze(0) if x.dim() == 2 else y.view(-1)

def pad_or_trim(x: torch.Tensor, target_len: int, align: str = "left") -> torch.Tensor:
    """
    Pad (zeros) or trim along time to exactly `target_len`.
    align: 'left' (keep beginning), 'center', or 'right' (keep ending) when trimming.
    """
    x3 = ensure_3d(x)
    B, C, T = x3.shape
    if T == target_len:
        return x
    if T < target_len:
        pad = target_len - T
        pad_left = 0
        pad_right = pad
        return torch.nn.functional.pad(x3, (0, pad_right), mode="constant", value=0.0) \
            if x.dim() == 3 else pad_or_trim(x3, target_len, align)
    # Trim
    if align == "center":
        start = max(0, (T - target_len) // 2)
    elif align == "right":
        start = max(0, T - target_len)
    else:
        start = 0
    return x3[..., start:start + target_len] if x.dim() == 3 else x3[..., start:start + target_len].squeeze(0)

def match_length(a: torch.Tensor, b: torch.Tensor, align: str = "left") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make two waveforms the same length by padding/trimming the *second* to the first.
    """
    a3 = ensure_3d(a); b3 = ensure_3d(b)
    Ta, Tb = a3.shape[-1], b3.shape[-1]
    if Ta == Tb:
        return a, b
    return a, pad_or_trim(b, target_len=Ta, align=align)

def random_time_crop_pair(
    xs: List[torch.Tensor], crop_samples: int, rng: Optional[torch.Generator] = None
) -> List[torch.Tensor]:
    """
    Take the same random crop across a list of aligned signals. Inputs can be [T]/[C,T]/[B,C,T].
    """
    xs3 = [ensure_3d(x) for x in xs]
    T = min(x.shape[-1] for x in xs3)
    if crop_samples >= T:
        return [pad_or_trim(x, crop_samples) for x in xs3]
    g = rng if rng is not None else torch.Generator(device=xs3[0].device)
    start = int(torch.randint(0, T - crop_samples + 1, (1,), generator=g).item())
    return [x[..., start:start + crop_samples] for x in xs3]

def random_gain_db(x: torch.Tensor, min_db: float = -3.0, max_db: float = +3.0,
                   per_channel: bool = False, rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Apply random gain in dB. Uniform in [min_db, max_db].
    """
    x3 = ensure_3d(x)
    B, C, T = x3.shape
    g = rng if rng is not None else torch.Generator(device=x3.device)
    if per_channel:
        gains = (min_db + (max_db - min_db) * torch.rand((B, C, 1), generator=g, device=x3.device, dtype=x3.dtype))
    else:
        gains = (min_db + (max_db - min_db) * torch.rand((B, 1, 1), generator=g, device=x3.device, dtype=x3.dtype))
    scale = (10.0 ** (gains / 20.0))
    y = x3 * scale
    return y if x.dim() == 3 else y.squeeze(0) if x.dim() == 2 else y.view(-1)

def is_silent(x: torch.Tensor, threshold_db: float = -60.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns a boolean mask per-item (shape [B, 1] for [B,C,T], or scalar for single item)
    indicating RMS dB < threshold_db.
    """
    from .metrics import rms_db
    x3 = ensure_3d(x)
    # RMS over time and channels jointly
    r_db = rms_db(x3, dim=(-1, -2), keepdim=True)  # [B,1,1]
    silent = (r_db < threshold_db)
    return silent.squeeze(-1)  # [B,1]
