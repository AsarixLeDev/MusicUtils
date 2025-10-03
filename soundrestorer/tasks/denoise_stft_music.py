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
from torch._dynamo import disable

@disable  # ensure this runs outside the compiled graph
def to_float(val):
    try:
        return float(val.detach().cpu().item())
    except Exception:
        try:
            return float(val)
        except Exception:
            return float("nan")


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


import torch

def _variant_from_head(
    head: torch.Tensor,          # (B,2,F,T) from the model head
    variant: str,                # 'complex_delta' | 'complex' | 'mag_sigmoid' | 'mag' | 'mag_delta1'
    cfg,                         # task cfg: may have mask_floor, mask_limit, clamp_mask_tanh
    compute_stats: bool = False  # set True only if you want scalar stats (will be skipped under compile)
) -> tuple[torch.Tensor, dict]:
    """
    Map raw head outputs -> complex mask M = R + j I  (shape (B,2,F,T)).

    Variants
    --------
    - 'complex_delta' (alias: 'delta1', 'delta1_complex'):
        R = 1 + Mr,  I = Mi  (identity-anchored complex mask; yhat≈noisy at init)
    - 'complex' / 'plain':
        R = Mr,      I = Mi  (plain complex mask; often attenuates early if head starts near 0)
    - 'mag_sigmoid':
        Mag = 1 + (sigmoid(Mr) - 0.5) * 2  in [0, 2] ;  I = 0
    - 'mag':
        Mag = sqrt(Mr^2 + Mi^2) ; I = 0
    - 'mag_delta1':
        Mag = 1 + sqrt(Mr^2 + Mi^2) ; I = 0

    Post mapping guards (optional in cfg):
      - mask_floor        : minimum magnitude (>=0)   – floors |M|
      - mask_limit        : maximum magnitude (>0)    – caps   |M|
      - clamp_mask_tanh   : component clamp via tanh  – clamps R and I to [-c, +c]

    Returns
    -------
    mask_b2ft : torch.Tensor (B,2,F,T)
    stats     : dict[str, float] (empty when compiling or compute_stats=False)
    """
    Mr = head[:, 0]
    Mi = head[:, 1]
    v  = str(variant).lower()

    # ---- mapping ----
    if v in ("complex_delta", "delta1", "delta1_complex"):
        # identity-anchored complex mask
        R = 1.0 + Mr
        I = Mi
    elif v in ("complex", "plain"):
        R = Mr
        I = Mi
    elif v in ("mag_sigmoid", "mag_sigm1"):
        Mag = 1.0 + (torch.sigmoid(Mr) - 0.5) * 2.0  # [0, 2]
        R = Mag
        I = torch.zeros_like(Mr)
    elif v == "mag_delta1":
        Mag = 1.0 + torch.sqrt(Mr * Mr + Mi * Mi + 1e-8)
        R = Mag
        I = torch.zeros_like(Mr)
    elif v == "mag":
        Mag = torch.sqrt(Mr * Mr + Mi * Mi + 1e-8)
        R = Mag
        I = torch.zeros_like(Mr)
    else:
        # sensible default: plain complex
        R = Mr
        I = Mi

    # ---- optional component clamp via tanh (before magnitude guards) ----
    c_tanh = float(getattr(cfg, "clamp_mask_tanh", 0.0) or 0.0)
    if c_tanh > 0.0:
        R = torch.tanh(R) * c_tanh
        I = torch.tanh(I) * c_tanh

    # ---- magnitude floor/limit (on |M|) ----
    mag = torch.sqrt(R * R + I * I + 1e-12)

    # floor first (push small magnitudes up)
    floor = float(getattr(cfg, "mask_floor", 0.0) or 0.0)
    if floor > 0.0:
        # scale samples with mag < floor up to exactly 'floor'
        scale_up = torch.clamp(floor / mag, max=1e6)  # avoid INF
        mask = mag < floor
        if mask.any():
            R = torch.where(mask, R * scale_up, R)
            I = torch.where(mask, I * scale_up, I)
            mag = torch.sqrt(R * R + I * I + 1e-12)

    # then cap (push large magnitudes down)
    limit = float(getattr(cfg, "mask_limit", 0.0) or 0.0)
    if limit > 0.0:
        scale_dn = torch.clamp(limit / mag, max=1.0)
        R = R * scale_dn
        I = I * scale_dn
        # (no need to recompute 'mag' unless you want it in stats)

    mask_b2ft = torch.stack([R, I], dim=1)

    # ---- optional stats (avoid graph breaks under torch.compile) ----
    stats: dict[str, float] = {}
    if compute_stats:
        try:
            import torch._dynamo as dynamo
            compiling = bool(getattr(dynamo, "is_compiling", lambda: False)())
        except Exception:
            compiling = False
        if not compiling:
            with torch.no_grad():
                # use detach().cpu() and safe float conversion
                def _sf(x):
                    try:
                        return float(x.detach().cpu().mean().item())
                    except Exception:
                        return float("nan")
                stats["mask_R_mean"] = _sf(R)
                stats["mask_I_mean"] = _sf(I)
                stats["mask_mag_mean"] = _sf(torch.sqrt(R * R + I * I + 1e-12))
    return mask_b2ft, stats



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
        win = torch.hann_window(self.cfg.win_length, device=noisy_bt.device, dtype=torch.float32)
        X = torch.stft(
            noisy_bt.to(torch.float32), n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length, win_length=self.cfg.win_length,
            window=win, center=True, return_complex=True
        )  # (B,F,T) complex64

        # real/imag for the model input (B,2,F,T)
        Xri = torch.view_as_real(X).permute(0, 3, 1, 2).contiguous()  # (B,2,F,T), float32

        # --- model head ---
        head = self.model(Xri)  # (B,2,F,T) float
        Mri, mstats = _variant_from_head(  # your mapper, e.g. 'complex_delta'
            head, self.cfg.mask_variant, self.cfg, compute_stats=False
        )
        # right after: Mri, mstats = _variant_from_head(...)
        if not hasattr(self, "_mask_probe_done") and (self.training is True):
            with torch.no_grad():
                R = Mri[:, 0];
                I = Mri[:, 1]
                devR = (R - 1.0).abs().mean().item()
                devI = I.abs().mean().item()
                print(f"[mask] mean|R-1|={devR:.4e} | mean|I|={devI:.4e}")
            self._mask_probe_done = True

        # --- build complex mask and apply (stay complex thereafter) ---
        # match dtypes to X.real; keep complex64 for ISTFT
        R = Mri[:, 0].to(X.real.dtype)  # (B,F,T) float
        I = Mri[:, 1].to(X.real.dtype)  # (B,F,T) float
        M = torch.complex(R, I)  # (B,F,T) complex
        Shat = M * X  # (B,F,T) complex

        # --- ISTFT ---
        yhat = torch.istft(
            Shat, n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length, win_length=self.cfg.win_length,
            window=win, center=True, length=T, return_complex=False
        )  # (B,T) float32

        if self.cfg.safe_unity_fallback and not torch.isfinite(yhat).all():
            yhat = noisy_bt.clone()

        out = {"yhat": yhat, "R": R, "I": I}
        if self.cfg.return_debug:
            out["debug"] = {"mask_stats": mstats}
        return out

