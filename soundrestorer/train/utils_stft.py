from __future__ import annotations

import torch

__all__ = ["stft_pair", "istft_from", "hann_window"]


def hann_window(n_fft: int, device: str | torch.device = "cpu", dtype=torch.float32):
    return torch.hann_window(n_fft, device=device, dtype=dtype)


def stft_pair(x: torch.Tensor, win: torch.Tensor, n_fft=1024, hop=256):
    """
    x: (B,T) or (B,C,T) waveform in [-1,1] float
    Returns:
      X   : complex STFT (B,F,T) if mono, or (B*C,F,T) if you flatten channels yourself upstream
      Xri : real/imag stack (B,2,F,T)
    NOTE: keep complex math in float32 for bf16 compatibility.
    """
    if x.dim() == 3:
        # merge channels into batch if needed; caller can handle multi-channel explicitly
        B, C, T = x.shape
        x = x.reshape(B * C, T)
    X = torch.stft(x.float(), n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True, center=True)
    Xri = torch.stack([X.real, X.imag], dim=1)  # (B,2,F,T)
    return X, Xri


def istft_from(X: torch.Tensor, win: torch.Tensor, length: int, n_fft=1024, hop=256):
    """
    X: complex STFT (B,F,T)
    Returns waveform (B,T)
    """
    return torch.istft(X, n_fft=n_fft, hop_length=hop, window=win,
                       center=True, length=length)
