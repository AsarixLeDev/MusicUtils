# -*- coding: utf-8 -*-
"""
DenoiseSTFTMusic
- Compute STFT of noisy waveform
- Feed magnitude (or RI) to model that predicts a mask
- Apply mask (mag/complex/delta variants)
- iSTFT back to waveform

Compatible with your trainer: outputs["yhat"] is waveform (B, T).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math, torch, torch.nn as nn
import torchaudio

# ---------------- cfg ----------------

@dataclass
class DenoiseSTFTCfg:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    center: bool = True
    window: str = "hann"

    # mask behavior
    mask_variant: str = "mag_sigmoid"  # ["mag_sigmoid", "mag", "mag_delta1", "delta1", "complex"]
    mask_floor: float = 0.30
    mask_limit: float = 1.80
    clamp_mask_tanh: float = 0.0  # optional: extra tanh clamp radius; 0.0 = off
    safe_unity_fallback: bool = True  # if mask becomes NaN/Inf, use 1.0

    # numerical
    eps: float = 1e-8
    return_debug: bool = False

# ------------- helpers -------------

def _get_window(name: str, n_fft: int, device, dtype):
    name = (name or "hann").lower()
    if name == "hann":
        return torch.hann_window(n_fft, device=device, dtype=dtype)
    if name == "hamming":
        return torch.hamming_window(n_fft, device=device, dtype=dtype)
    # default
    return torch.hann_window(n_fft, device=device, dtype=dtype)

def _stft(y: torch.Tensor, cfg: DenoiseSTFTCfg) -> torch.Tensor:
    win = _get_window(cfg.window, cfg.n_fft, y.device, torch.float32)
    return torch.stft(
        y.to(torch.float32), n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        window=win, return_complex=True, center=cfg.center, pad_mode="reflect"
    )

def _istft(spec: torch.Tensor, cfg: DenoiseSTFTCfg, length: int) -> torch.Tensor:
    win = _get_window(cfg.window, cfg.n_fft, spec.device, torch.float32)
    y = torch.istft(
        spec.to(torch.complex64), n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        window=win, center=cfg.center, length=length
    )
    return y

def _apply_mag_mask(spec: torch.Tensor, mask: torch.Tensor, cfg: DenoiseSTFTCfg) -> torch.Tensor:
    mag = spec.abs()
    pha = torch.angle(spec)  # noqa: E999 (ok at runtime)
    mag_hat = mag * mask
    return torch.polar(mag_hat, pha)

def _safe_mask(mask: torch.Tensor, cfg: DenoiseSTFTCfg) -> torch.Tensor:
    if cfg.clamp_mask_tanh and cfg.clamp_mask_tanh > 0:
        mask = torch.tanh(mask / cfg.clamp_mask_tanh) * cfg.clamp_mask_tanh
    # Always enforce [floor, limit] for magnitude-like masks
    return mask.clamp(min=cfg.mask_floor, max=cfg.mask_limit)

def _variant_from_head(head_out: torch.Tensor, variant: str, cfg: DenoiseSTFTCfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Convert raw head_out (B, C, F, T) to complex or magnitude mask,
    returning (complex_mask or mag_mask, stats).
    """
    stats = {}
    v = (variant or "mag_sigmoid").lower()
    B, C, F, TT = head_out.shape

    if v in ("mag_sigmoid", "mag", "mag_delta1", "delta1"):
        # Produce a magnitude mask in [mask_floor, mask_limit]
        if v == "mag_sigmoid":
            m = torch.sigmoid(head_out[:, 0:1])
            mask = cfg.mask_floor + (cfg.mask_limit - cfg.mask_floor) * m
        elif v == "mag_delta1":
            # 1 + residual, then bound
            mask = 1.0 + head_out[:, 0:1]
            mask = _safe_mask(mask, cfg)
        elif v == "delta1":
            # legacy delta around 1.0 but still magnitude-only
            mask = 1.0 + torch.tanh(head_out[:, 0:1])
            mask = _safe_mask(mask, cfg)
        else:
            # "mag": unbounded -> squash via sigmoid to stay stable
            m = torch.sigmoid(head_out[:, 0:1])
            mask = cfg.mask_floor + (cfg.mask_limit - cfg.mask_floor) * m
        stats["mask_min"] = float(mask.min().item())
        stats["mask_max"] = float(mask.max().item())
        return mask, stats

    # complex mask (M_re, M_im) with optional amplitude clamp
    if C < 2:
        # upgrade to 2 channels by duplication if needed
        z = torch.zeros_like(head_out[:, 0:1])
        head_out = torch.cat([head_out[:, 0:1], z], dim=1)

    M_re = head_out[:, 0]
    M_im = head_out[:, 1]

    if v == "complex_delta1":
        M_re = 1.0 + M_re  # residual around 1+0j

    M = torch.complex(M_re, M_im)
    # soft clamp amplitude
    amp = torch.abs(M)
    amp_clamped = amp.clamp(min=cfg.mask_floor, max=cfg.mask_limit)
    M = M * (amp_clamped / (amp + cfg.eps))

    stats["mask_amp_min"] = float(amp_clamped.min().item())
    stats["mask_amp_max"] = float(amp_clamped.max().item())
    return M, stats

# ------------- main task -------------

class DenoiseSTFTMusic(nn.Module):
    def __init__(self, model: nn.Module, args: Dict[str, Any]):
        super().__init__()
        # model: any 2D UNet-like that maps (B, C_in=1, F, T) -> (B, C_out, F, T)
        self.model = model
        self.cfg = DenoiseSTFTCfg(
            n_fft           = int(args.get("n_fft", 1024)),
            hop_length     = int(args.get("hop_length", 256)),
            win_length     = int(args.get("win_length", 1024)),
            center         = bool(args.get("center", True)),
            window         = str(args.get("window", "hann")),
            mask_variant   = str(args.get("mask_variant", "mag_sigmoid")),
            mask_floor     = float(args.get("mask_floor", 0.30)),
            mask_limit     = float(args.get("mask_limit", 1.80)),
            clamp_mask_tanh= float(args.get("clamp_mask_tanh", 0.0)),
            safe_unity_fallback = bool(args.get("safe_unity_fallback", True)),
            return_debug   = bool(args.get("return_debug", False)),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch["noisy"]: (B, T) or (B, 1, T)
        Return dict with "yhat": (B, T)
        """
        noisy = batch["noisy"]
        if noisy.dim() == 2:
            B, T = noisy.shape
            y = noisy
        elif noisy.dim() == 3 and noisy.size(1) == 1:
            B, _, T = noisy.shape
            y = noisy[:, 0]
        else:
            raise ValueError("Expected noisy as (B, T) or (B, 1, T)")

        S = _stft(y, self.cfg)          # (B, F, TT) complex
        mag = torch.abs(S)
        # Log1p magnitude input (stable across levels)
        x_in = torch.log1p(mag).unsqueeze(1)  # (B,1,F,TT)

        head = self.model(x_in)                 # (B, C_out, F, TT)
        mask_pred, mstats = _variant_from_head(head, self.cfg.mask_variant, self.cfg)

        if torch.is_complex(mask_pred):
            Shat = S * mask_pred
        else:
            # magnitude mask; keep noisy phase
            pha = torch.angle(S)
            mag_hat = mag * mask_pred
            Shat = torch.polar(mag_hat, pha)

        yhat = _istft(Shat, self.cfg, length=T)  # (B, T)

        if self.cfg.safe_unity_fallback:
            # If something went NaN/Inf, fallback to identity
            bad = torch.isnan(yhat) | torch.isinf(yhat)
            if bad.any():
                yhat = y.clone()

        out = {"yhat": yhat}
        if self.cfg.return_debug:
            out["debug"] = {
                "mask_stats": mstats,
            }
        return out
