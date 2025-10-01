# tests/test_losses.py
import math
import torch
import pytest

from soundrestorer.utils.metrics import si_sdr_db

# Local LSD (failsafe if not provided by utils.metrics)
def _lsd_db(a: torch.Tensor, b: torch.Tensor, n_fft=2048, hop=512, eps=1e-8):
    from soundrestorer.utils.signal import stft_complex
    A = stft_complex(a, n_fft=n_fft, hop_length=hop)
    B = stft_complex(b, n_fft=n_fft, hop_length=hop)
    AdB = (A.abs().clamp_min(eps)).log10().mul(20.0)
    BdB = (B.abs().clamp_min(eps)).log10().mul(20.0)
    d = (AdB - BdB).pow(2).mean(dim=-2).sqrt().mean()
    return float(d.item())

def test_sisdr_pos_identity_is_zero_and_positive_when_worse():
    torch.manual_seed(0)
    x = torch.randn(1,1,16000) * 0.1
    y = x.clone()
    # "Positive" loss = clamp(-SI-SDR, 0+) or a dedicated loss module.
    si = float(si_sdr_db(y, x)[0].item())  # ~ large positive
    # define loss: max(0, -SI) to force >=0 and 0 at identity (SI >= 0 typically)
    loss_id = max(0.0, -si)
    assert abs(loss_id) < 1e-6

    # Degrade
    y2 = x + 0.2 * torch.randn_like(x)
    si2 = float(si_sdr_db(y2, x)[0].item())
    loss2 = max(0.0, -si2)
    assert loss2 >= 0.0
    assert loss2 > loss_id

def test_lsd_stability():
    torch.manual_seed(0)
    x = torch.randn(1,1,16000) * 0.05
    y = x.clone()
    l0 = _lsd_db(x, y)
    assert l0 < 1e-6  # identical spectra

    z = x + 0.1 * torch.randn_like(x)
    l1 = _lsd_db(x, z)
    assert l1 > l0 + 1e-4
