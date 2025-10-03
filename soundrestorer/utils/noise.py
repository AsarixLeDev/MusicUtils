from __future__ import annotations

import math
import random
import torch

from typing import Optional


def _rms(x, dim=-1, keepdim=True):
    return torch.sqrt(torch.clamp(torch.mean(x ** 2, dim=dim, keepdim=keepdim), min=1e-12))


# add helper
def _to_bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3: return x.mean(dim=1)
    raise RuntimeError(f"mix_at_snr expects 1/2/3D, got {tuple(x.shape)}")


def mix_at_snr(clean_bt: torch.Tensor, noise_bt: torch.Tensor, snr_db: torch.Tensor | float,
               peak: Optional[float] = 0.98, eps: float = 1e-12) -> torch.Tensor:
    """
    clean_bt, noise_bt: (B,T) float32/-1..1
    snr_db:             (B,) or scalar
    peak:               float -> peak-normalize mixture; None -> no re-peak normalization
    """
    clean = clean_bt
    noise = noise_bt

    if isinstance(snr_db, (int, float)):
        snr_db = torch.full((clean.shape[0],), float(snr_db), device=clean.device, dtype=clean.dtype)

    # energies
    Pc = torch.mean(clean**2, dim=-1, keepdim=True).clamp_min(eps)
    Pn = torch.mean(noise**2, dim=-1, keepdim=True).clamp_min(eps)

    # scale noise for target SNR
    alpha = torch.sqrt(Pc / (Pn * (10.0 ** (snr_db.view(-1, 1) / 10.0))))
    mix = clean + alpha * noise  # (B,T)

    if peak is not None:
        # re-peak normalize (cap at 'peak', do not boost if already below)
        maxabs = mix.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        g = torch.clamp(torch.tensor(float(peak), device=mix.device, dtype=mix.dtype) / maxabs, max=1.0)
        mix = mix * g

    return mix


def _colored_noise(color: str, shape, device):
    """
    'white': flat
    'pink':  1/sqrt(f)
    'brown': 1/f
    'violet': f
    Implemented by shaping in FFT domain.
    """
    B, T = shape
    x = torch.randn(B, T, device=device)
    if color == "white": return x
    # FFT
    X = torch.fft.rfft(x, n=T, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0) + 1e-12  # avoid div0
    if color == "pink":
        H = 1.0 / torch.sqrt(freqs)
    elif color == "brown":
        H = 1.0 / freqs
    elif color == "violet":
        H = torch.sqrt(freqs)
    else:
        return x
    H = H.to(device=device, dtype=X.real.dtype)
    X = X * H  # broadcast on last dim
    y = torch.fft.irfft(X, n=T, dim=-1)
    # normalize rms
    y = y / (_rms(y))
    return y


def _bandpass_white(shape, device, sr, f_lo, f_hi):
    B, T = shape
    x = torch.randn(B, T, device=device)
    X = torch.fft.rfft(x, n=T, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sr).to(device)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X = X * mask
    y = torch.fft.irfft(X, n=T, dim=-1)
    y = y / (_rms(y))
    return y


def _hum(shape, device, sr, f0=None, harmonics=(2, 3, 4, 5), drift_cents=5.0):
    B, T = shape
    if f0 is None:
        f0 = random.choice([50.0, 60.0])
    t = torch.arange(T, device=device).unsqueeze(0) / sr
    # small random detune
    det = 2.0 ** (torch.randn(B, 1, device=device) * (drift_cents / 1200.0))
    base = 2 * math.pi * f0
    y = torch.zeros(B, T, device=device)
    amps = torch.rand(B, 1 + len(harmonics), device=device)
    amps = amps / (amps.sum(dim=-1, keepdim=True) + 1e-6)
    # fundamental
    y += amps[:, 0:1] * torch.sin(det * base * t)
    # harmonics
    for i, h in enumerate(harmonics):
        y += amps[:, i + 1:i + 2] * torch.sin(det * (base * h) * t)
    # light AM
    am = 1.0 + 0.1 * torch.sin(2 * math.pi * random.uniform(0.1, 1.0) * t)
    y = y * am
    return y / (_rms(y))


def _clicks(shape, device, density=0.001, click_len=32, decay=0.9):
    B, T = shape
    y = torch.zeros(B, T, device=device)
    num = max(1, int(T * density))
    for b in range(B):
        pos = torch.randint(0, T - click_len, (num,), device=device)
        for p in pos:
            amp = torch.randn(1, device=device).abs().clamp_(0.1, 1.0)
            k = torch.arange(click_len, device=device)
            impulse = amp * (decay ** k)
            y[b, p:p + click_len] += impulse
    return y / (_rms(y))


class NoiseFactory:
    """
    Sample various procedural noises. Configure relative weights per type.
    """

    def __init__(self, sr, cfg):
        self.sr = int(sr)
        # weights
        self.weights = cfg.get("weights", {
            "white": 1.0, "pink": 1.0, "brown": 0.5,
            "band": 1.0, "hum": 0.5, "clicks": 0.3
        })
        # band params
        self.band_lo = cfg.get("band_lo", 100.0)
        self.band_hi = cfg.get("band_hi", 12000.0)
        self.band_min_bw = cfg.get("band_min_bw", 200.0)

    def _one(self, shape, device):
        kinds, ws = zip(*self.weights.items())
        ws = torch.tensor(ws, dtype=torch.float32)
        ws = (ws / ws.sum()).tolist()
        kind = random.choices(kinds, weights=ws, k=1)[0]
        if kind in ("white", "pink", "brown", "violet"):
            return _colored_noise(kind, shape, device)
        if kind == "band":
            f1 = random.uniform(self.band_lo, max(self.band_lo, self.band_hi - self.band_min_bw))
            f2 = random.uniform(f1 + self.band_min_bw, self.band_hi)
            return _bandpass_white(shape, device, self.sr, f1, f2)
        if kind == "hum":
            return _hum(shape, device, self.sr)
        if kind == "clicks":
            return _clicks(shape, device)
        # fallback
        return _colored_noise("white", shape, device)

    def sample(self, T, device):
        return self._one((1, T), device=device).squeeze(0)

    def sample_batch(self, B, T, device):
        return self._one((B, T), device=device)
