from __future__ import annotations

import torch

# Public API
__all__ = [
    "si_sdr_ratio_loss",
    "noise_floor_weight",
    "mel_filterbank",
    "log_mel_L1",
    "highband_mag_L1",
]


# ---- SI-SDR (ratio form, positive, →0 as it improves) ----
def si_sdr_ratio_loss(y: torch.Tensor, x: torch.Tensor, min_db: float = -55.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Positive SI-SDR 'ratio' loss = ||e||^2 / ||s||^2 (≥0). Lower is better.
    - Robust to silence via min_db gating on the clean RMS.
    - Works with (B,T) or (B,C,T) inputs; multi-channel is averaged for SI-SDR.
    """
    if y.dim() == 3:
        y = y.mean(dim=1)
        x = x.mean(dim=1)

    x0 = x - x.mean(dim=-1, keepdim=True)
    y0 = y - y.mean(dim=-1, keepdim=True)

    s = (torch.sum(y0 * x0, dim=-1, keepdim=True) * x0) / (torch.sum(x0 * x0, dim=-1, keepdim=True) + eps)
    e = y0 - s

    num = torch.sum(e * e, dim=-1) + eps
    den = torch.sum(s * s, dim=-1) + eps
    ratio = num / den  # ≥ 0

    rms = torch.sqrt((x0 * x0).mean(dim=-1) + eps)
    rms_db = 20.0 * torch.log10(rms + eps)
    valid = (rms_db > min_db)

    if valid.any():
        return ratio[valid].mean()
    return torch.zeros((), device=y.device)


# ---- Noise-floor weighting (punish residual energy where clean is quiet) ----
def noise_floor_weight(clean_mag: torch.Tensor, nf_db: float = -45.0) -> torch.Tensor:
    """
    clean_mag: |STFT(clean)|, shape [B, F, T]
    Returns weights w in [0,1], larger where clean is quiet.
    """
    cm = clean_mag
    cm_max = cm.amax(dim=(1, 2), keepdim=True) + 1e-8
    cm_db = 20.0 * torch.log10(torch.clamp(cm / cm_max, min=1e-8))
    # Steeper sigmoid makes a crisp transition near nf_db
    w = torch.sigmoid(4.0 * (nf_db - cm_db))
    return w


# ---- Mel utilities and perceptual losses ----
def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Returns triangular mel filterbank matrix [M, F] that maps linear |STFT|^2 to M mel bands.
    """
    fft_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, steps=fft_bins, device=device, dtype=dtype)

    mmin = _hz_to_mel(torch.tensor(0.0, device=device, dtype=dtype))
    mmax = _hz_to_mel(torch.tensor(sr / 2, device=device, dtype=dtype))
    mgrid = torch.linspace(mmin, mmax, steps=n_mels + 2, device=device, dtype=dtype)
    hz = _mel_to_hz(mgrid)

    fb = torch.zeros(n_mels, fft_bins, device=device, dtype=dtype)
    for i in range(n_mels):
        f_l, f_c, f_r = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - f_l) / (f_c - f_l + 1e-8)
        right = (f_r - freqs) / (f_r - f_c + 1e-8)
        fb[i] = torch.clamp(torch.minimum(left, right), min=0.0)
    return fb  # [M, F]


def log_mel_L1(y: torch.Tensor, x: torch.Tensor, win: torch.Tensor, sr: int, n_mels: int) -> torch.Tensor:
    """
    L1 distance between log-mel power spectra (B, M, T). Uses a higher floor to avoid log underflow.
    """
    # Lazy import to avoid circularity
    from .utils_stft import stft_pair
    Y, _ = stft_pair(y, win)
    X, _ = stft_pair(x, win)
    Ypow = (Y.abs().float() ** 2)
    Xpow = (X.abs().float() ** 2)

    fb = mel_filterbank(sr=sr, n_fft=win.shape[0], n_mels=n_mels, device=y.device, dtype=torch.float32)
    Ym = torch.einsum("mf,bft->bmt", fb, Ypow)
    Xm = torch.einsum("mf,bft->bmt", fb, Xpow)

    # robust log floor
    logY = torch.log10(Ym + 1e-4)
    logX = torch.log10(Xm + 1e-4)
    return torch.mean(torch.abs(logY - logX))


def highband_mag_L1(y: torch.Tensor, x: torch.Tensor, win: torch.Tensor, sr: int, cutoff_khz: float) -> torch.Tensor:
    """
    Weighted L1 on magnitude above ~cutoff_khz, softly ramped to protect 'air'/brightness.
    """
    from .utils_stft import stft_pair
    Y, _ = stft_pair(y, win)
    X, _ = stft_pair(x, win)
    Ymag = Y.abs().float()
    Xmag = X.abs().float()

    F = Ymag.shape[1]
    freqs = torch.linspace(0, sr / 2, steps=F, device=y.device, dtype=torch.float32)
    w = torch.sigmoid((freqs - cutoff_khz * 1000.0) / 500.0).view(1, F, 1)  # (1,F,1)
    return torch.mean(w * torch.abs(Ymag - Xmag))
