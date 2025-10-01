# -*- coding: utf-8 -*-
# soundrestorer/metrics/common.py
"""
Canonical audio/signal helpers used across tools, metrics, and callbacks.

All helpers here are shape-safe and torchscript-friendly where possible.

Conventions:
- Audio tensors are channel-first [C, T] (torchaudio default). We convert to mono
  as [1, T] for most scalar metrics.
- All SI/ratio computations are scale-invariant unless stated.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torchaudio

from ..utils.audio import to_mono
from ..utils.metrics import si_sdr_db

_EPS = EPS = 1e-10


# -----------------------------
# Basic tensor utilities
# -----------------------------


def match_len(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop both tensors on last dim to min length."""
    n = min(a.size(-1), b.size(-1))
    return a[..., :n], b[..., :n]


def rms(x: torch.Tensor) -> torch.Tensor:
    """RMS over time axis; keeps batch/channel where present."""
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1,T]
    return torch.sqrt((x ** 2).mean(dim=-1) + EPS)


def power_ratio_db(num: torch.Tensor, den: torch.Tensor) -> float:
    """10*log10( ||num||^2 / ||den||^2 )."""
    n = torch.sum(num ** 2) + EPS
    d = torch.sum(den ** 2) + EPS
    return float(10.0 * torch.log10(n / d))


# -----------------------------
# SI‑SDR & SNR family
# -----------------------------

def snr_db(noisy: torch.Tensor, clean: torch.Tensor) -> float:
    """
    Linear SNR in dB: 10*log10( ||clean||^2 / ||noisy-clean||^2 ).
    """
    noisy_m = to_mono(noisy).to(torch.float32)
    clean_m = to_mono(clean).to(torch.float32)
    noisy_m, clean_m = match_len(noisy_m, clean_m)
    noise = noisy_m - clean_m
    return power_ratio_db(clean_m, noise)


def si_delta_db(yhat: torch.Tensor, noisy: torch.Tensor, clean: torch.Tensor) -> float:
    """
    ΔSI‑SDR: si_sdr(yhat,clean) - si_sdr(noisy,clean).
    """
    return si_sdr_db(yhat, clean) - si_sdr_db(noisy, clean)


# -----------------------------
# Errors and diagnostics
# -----------------------------
def rel_error_db(target: torch.Tensor, approx: torch.Tensor) -> float:
    """
    Relative error in dB:
        20*log10( rms(target-approx) / rms(target) )
    More negative is better; e.g., -40 dB means very tight.
    """
    t = to_mono(target)
    a = to_mono(approx)
    t, a = match_len(t, a)
    e = torch.sqrt(((t - a) ** 2).mean(dim=-1) + EPS).mean()
    d = torch.sqrt((t ** 2).mean(dim=-1) + EPS).mean()
    return float(20.0 * torch.log10((e + EPS) / (d + EPS)))


def silence_fraction(x: torch.Tensor, amp_thr: float = 1e-4) -> float:
    """Fraction of |x| < amp_thr (mono‑reduced)."""
    m = to_mono(x)
    return float((m.abs() < amp_thr).float().mean().item())


def clip_fraction(x: torch.Tensor, clip_thr: float = 1.0) -> float:
    """Fraction of |x| > clip_thr (mono‑reduced)."""
    m = to_mono(x)
    return float((m.abs() > clip_thr).float().mean().item())


# -----------------------------
# I/O helpers
# -----------------------------
def resample_if_needed(x: torch.Tensor, sr_from: int, sr_to: Optional[int]) -> torch.Tensor:
    """Resample x from sr_from to sr_to if sr_to is provided and differs."""
    if sr_to is not None and sr_from != sr_to:
        return torchaudio.functional.resample(x, sr_from, sr_to)
    return x


def match_lengths(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop both to the same min length (time last dim)."""
    n = min(a.size(-1), b.size(-1))
    return a[..., :n], b[..., :n]


def resample_if_needed(x: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
    if sr_from == sr_to:
        return x
    return torchaudio.functional.resample(x, sr_from, sr_to)


# -----------------------------
#   SI-SDR (dB) + error ratio
# -----------------------------
def _si_parts(est: torch.Tensor, ref: torch.Tensor, eps: float = _EPS):
    # zero-mean along time
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_pow = (ref ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / ref_pow) * ref
    e = est - s_target
    return s_target, e


@torch.no_grad()
def si_sdr_error_ratio(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Positive error metric: ||e||^2 / (||s_target||^2 + eps). Returns [B] (or [1]).
    → 0 when perfect (up to scale). Good 'loss-like' metric for eval dashboards.
    """
    est = to_mono(est).to(torch.float32)
    ref = to_mono(ref).to(torch.float32)
    est, ref = match_lengths(est, ref)
    s_t, e = _si_parts(est, ref)
    p_s = (s_t ** 2).sum(dim=-1) + _EPS
    p_e = (e ** 2).sum(dim=-1)
    return (p_e / p_s)


# -------------
#     SNR
# -------------
@torch.no_grad()
def snr_db(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SNR(x,y) in dB treating y as reference signal."""
    x = to_mono(x).to(torch.float32)
    y = to_mono(y).to(torch.float32)
    x, y = match_lengths(x, y)
    num = (y ** 2).sum(dim=-1)
    den = ((x - y) ** 2).sum(dim=-1) + _EPS
    return 10.0 * torch.log10((num + _EPS) / den)


# -----------------------------
#   Spectral metrics (LSD/MRSTFT)
# -----------------------------
def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int, center: bool = True):
    x = to_mono(x)  # make sure THIS returns [B,T], not [B,1,T]
    if x.dim() == 3 and x.size(1) == 1:
        x = x[:, 0]  # squeeze to [B,T]
    win_t = torch.hann_window(win, device=x.device, dtype=torch.float32)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_t.numel(),
                      window=win_t, center=center, return_complex=True).abs()


@torch.no_grad()
def lsd_db(yhat: torch.Tensor, clean: torch.Tensor,
           n_fft: int = 1024, hop: int = 256, win: int = 1024, center: bool = True) -> torch.Tensor:
    """
    Log-Spectral Distance (RMS in dB domain).
    Returns scalar per item [B].
    """
    Y = _stft_mag(yhat, n_fft, hop, win, center=center)
    T = _stft_mag(clean, n_fft, hop, win, center=center)
    dY = 20.0 * torch.log10(Y + _EPS)
    dT = 20.0 * torch.log10(T + _EPS)
    diff2 = (dY - dT) ** 2
    # RMS over TF, per item
    rms = torch.sqrt(diff2.mean(dim=(-2, -1)))
    return rms


@torch.no_grad()
def mrstft_metric(yhat: torch.Tensor, clean: torch.Tensor,
                  fft_sizes: List[int] = (1024, 2048, 512),
                  hops: List[int] = (256, 512, 128),
                  wins: List[int] = (1024, 2048, 512),
                  alpha: float = 0.5, beta: float = 0.5, center: bool = True) -> torch.Tensor:
    """
    Same math as MRSTFT loss, but no grad and averaged over resolutions.
    Returns [B] values (≥0, →0).
    """
    vals = []
    for n_fft, hop, win in zip(fft_sizes, hops, wins):
        Y = _stft_mag(yhat, n_fft, hop, win, center=center)
        T = _stft_mag(clean, n_fft, hop, win, center=center)
        diff = (Y - T).abs()
        l_mag = diff.mean(dim=(-2, -1))  # [B]
        # spectral convergence
        num = torch.linalg.vector_norm(diff, ord='fro', dim=(-2, -1))
        den = torch.linalg.vector_norm(T, ord='fro', dim=(-2, -1))
        sc = num / (den + _EPS)
        vals.append(alpha * l_mag + beta * sc)
    return torch.stack(vals, dim=0).mean(dim=0)  # [B]


# -----------------------------
#   STOI / PESQ (optional)
# -----------------------------
def _safe_import_stoi_pesq():
    stoi_fn = pesq_fn = None
    try:
        from pystoi import stoi as _stoi
        def stoi_fn(clean, proc, sr):
            # expects numpy, but we operate in torch—convert safely per item
            c = clean.detach().cpu().numpy()
            p = proc.detach().cpu().numpy()
            return float(_stoi(c, p, sr, extended=False))
    except Exception:
        pass

    try:
        # pip install pesq (ITU-T P.862 NB/WB). License notice applies.
        import pesq as _pesq
        def pesq_fn(clean, proc, sr):
            mode = "wb" if sr >= 16000 else "nb"
            c = clean.detach().cpu().numpy()
            p = proc.detach().cpu().numpy()
            return float(_pesq.pesq(sr, c, p, mode))
    except Exception:
        pass

    return stoi_fn, pesq_fn


STOI_FN, PESQ_FN = _safe_import_stoi_pesq()


@torch.no_grad()
def stoi_avg(yhat: torch.Tensor, clean: torch.Tensor, sr: int) -> Optional[torch.Tensor]:
    if STOI_FN is None: return None
    yhat = to_mono(yhat)
    clean = to_mono(clean)
    yhat, clean = match_lengths(yhat, clean)
    vals = []
    for i in range(yhat.size(0)):
        vals.append(STOI_FN(clean[i], yhat[i], sr))
    return torch.tensor(vals, dtype=torch.float32)


@torch.no_grad()
def pesq_avg(yhat: torch.Tensor, clean: torch.Tensor, sr: int) -> Optional[torch.Tensor]:
    if PESQ_FN is None: return None
    yhat = to_mono(yhat)
    clean = to_mono(clean)
    yhat, clean = match_lengths(yhat, clean)
    vals = []
    for i in range(yhat.size(0)):
        try:
            vals.append(PESQ_FN(clean[i], yhat[i], sr))
        except Exception:
            # PESQ throws on too-short clips; skip
            continue
    if not vals: return None
    return torch.tensor(vals, dtype=torch.float32)
