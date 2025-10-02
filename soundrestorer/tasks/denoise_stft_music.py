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
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn  # add near top if not present

from ..utils.audio import get_stft_window


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


def _stft(y: torch.Tensor, cfg: DenoiseSTFTCfg) -> torch.Tensor:
    win = get_stft_window(cfg.window, cfg.n_fft, y.device, torch.float32)
    return torch.stft(
        y.to(torch.float32), n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        window=win, return_complex=True, center=cfg.center, pad_mode="reflect"
    )


def _first_conv_in_channels(model: nn.Module) -> int:
    """Return in_channels of the first Conv2d found; default to 1 if none found."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return 1


def _istft(spec: torch.Tensor, cfg: DenoiseSTFTCfg, length: int) -> torch.Tensor:
    """
    spec: (B,F,T) complex (preferred) or (B,1,F,T) complex
    returns: (B,T) float32
    """
    if spec.dim() == 4 and spec.size(1) == 1:
        spec = spec[:, 0]  # -> (B,F,T)
    elif spec.dim() != 3:
        raise RuntimeError(f"ISTFT expects (B,F,T) or (B,1,F,T), got {tuple(spec.shape)}")

    win = get_stft_window(cfg.window, cfg.n_fft, spec.device, torch.float32)
    y = torch.istft(
        spec.to(torch.complex64),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=win,
        center=cfg.center,
        length=length,
        return_complex=False,
    )
    return y  # (B,T)


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


def _variant_from_head(head_out: torch.Tensor, variant: str, cfg: DenoiseSTFTCfg):
    """
    Convert raw head_out (B, C, F, T) to a mask of shape (B, F, T):
      - magnitude mask: real tensor in [mask_floor, mask_limit]
      - complex mask:   complex64 tensor with amplitude clamped to [floor, limit]
    """
    stats = {}
    v = (variant or "mag_sigmoid").lower()
    B, C, F, T = head_out.shape

    if v in ("mag_sigmoid", "mag", "mag_delta1", "delta1"):
        if v == "mag_sigmoid":
            m = torch.sigmoid(head_out[:, 0:1])  # (B,1,F,T)
            mask = cfg.mask_floor + (cfg.mask_limit - cfg.mask_floor) * m
        elif v == "mag_delta1":
            mask = 1.0 + head_out[:, 0:1]
            mask = mask.clamp(min=cfg.mask_floor, max=cfg.mask_limit)
        elif v == "delta1":
            mask = 1.0 + torch.tanh(head_out[:, 0:1])
            mask = mask.clamp(min=cfg.mask_floor, max=cfg.mask_limit)
        else:  # "mag"
            m = torch.sigmoid(head_out[:, 0:1])
            mask = cfg.mask_floor + (cfg.mask_limit - cfg.mask_floor) * m

        # SQUEEZE channel -> (B,F,T)
        mask = mask[:, 0]
        stats["mask_min"] = float(mask.min().item())
        stats["mask_max"] = float(mask.max().item())
        return mask, stats

    # complex variants -> (B,F,T) complex
    if C < 2:
        # duplicate real into imaginary if missing
        z = torch.zeros_like(head_out[:, 0:1])
        head_out = torch.cat([head_out[:, 0:1], z], dim=1)

    Mr = head_out[:, 0]
    Mi = head_out[:, 1]
    M = torch.complex(Mr, Mi)
    # amplitude clamp -> [floor, limit]
    amp = torch.abs(M)
    M = M * (amp.clamp(min=cfg.mask_floor, max=cfg.mask_limit) / (amp + cfg.eps))
    stats["mask_amp_min"] = float(amp.min().item())
    stats["mask_amp_max"] = float(amp.max().item())
    return M, stats


# ------------- main task -------------

class DenoiseSTFTMusic(nn.Module):
    def __init__(self, model: nn.Module, args: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.cfg = DenoiseSTFTCfg(
            n_fft=int(args.get("n_fft", 1024)),
            hop_length=int(args.get("hop_length", 256)),
            win_length=int(args.get("win_length", 1024)),
            center=bool(args.get("center", True)),
            window=str(args.get("window", "hann")),
            mask_variant=str(args.get("mask_variant", "mag_sigmoid")),
            mask_floor=float(args.get("mask_floor", 0.30)),
            mask_limit=float(args.get("mask_limit", 1.80)),
            clamp_mask_tanh=float(args.get("clamp_mask_tanh", 0.0)),
            safe_unity_fallback=bool(args.get("safe_unity_fallback", True)),
            return_debug=bool(args.get("return_debug", False)),
        )
        # NEW: detect what the model expects at its first conv
        self._in_ch = _first_conv_in_channels(self.model)

    def forward(self, batch: Dict[str, torch.Tensor] | List[torch.Tensor] | Tuple[torch.Tensor, ...]) -> Dict[
        str, torch.Tensor]:
        """
        Accept either:
          - dict: {"noisy": (B,T) or (B,1,T) or (B,C,T), ...}
          - tuple/list: [noisy, clean, ...] in classic dataset style
        Normalize 'noisy' to (B,T) float32 on the current device.
        """
        # -------------- get 'noisy' tensor --------------
        if isinstance(batch, dict):
            noisy = batch.get("noisy", None)
            if noisy is None:
                raise ValueError("batch dict must contain key 'noisy'")
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
            noisy = batch[0]
        else:
            raise ValueError("Batch must be a dict with 'noisy' or a list/tuple like [noisy, clean, ...]")

        import torch
        noisy = noisy.to(torch.float32)

        # -------------- normalize shape to (B,T) --------------
        if noisy.dim() == 1:
            # (T,) -> (1,T)
            noisy_bt = noisy.unsqueeze(0)
        elif noisy.dim() == 2:
            noisy_bt = noisy  # (B,T)
        elif noisy.dim() == 3:
            # (B,C,T) -> mono (B,T)
            noisy_bt = noisy.mean(dim=1)
        else:
            raise ValueError(f"Unexpected noisy shape {tuple(noisy.shape)}; expected (T)/(B,T)/(C,T)/(B,C,T)")

        B, T = noisy_bt.shape
        # ------------------------------------------------------

        # STFT
        S = _stft(noisy_bt, self.cfg)  # (B,F,T) complex
        mag = torch.abs(S)

        # choose input repr based on model in_channels
        if self._in_ch >= 2:
            x_in = torch.stack([S.real, S.imag], dim=1)  # (B,2,F,T)
        else:
            x_in = torch.log1p(mag).unsqueeze(1)  # (B,1,F,T)

        head = self.model(x_in)  # (B,C_out,F,T)
        M, mstats = _variant_from_head(head, self.cfg.mask_variant, self.cfg)

        if torch.is_complex(M):
            Shat = S * M  # (B,F,T) complex
        else:
            pha = torch.angle(S)
            mag_hat = mag * M  # (B,F,T)
            Shat = torch.polar(mag_hat, pha)

        yhat = _istft(Shat, self.cfg, length=T)  # (B,T)

        if self.cfg.safe_unity_fallback and not torch.isfinite(yhat).all():
            yhat = noisy_bt.clone()

        out = {"yhat": yhat}
        if self.cfg.return_debug:
            out["debug"] = {"mask_stats": mstats}
        return out

