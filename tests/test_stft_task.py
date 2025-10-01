# tests/test_stft_task.py
import pytest
import torch
from soundrestorer.utils.signal import stft_complex, istft_complex
from soundrestorer.utils.metrics import mse

def test_stft_istft_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(1,1,16000) * 0.05
    X = stft_complex(x, n_fft=1024, hop_length=256)
    y = istft_complex(X, n_fft=1024, hop_length=256, length=x.shape[-1])
    err = float(mse(x, y).item())
    assert err < 1e-6

@pytest.mark.skipif(
    pytest.importorskip("soundrestorer.tasks.denoise_stft", reason="STFT task not available") is None,
    reason="task missing",
)
def test_unity_mask_semantics():
    torch.manual_seed(0)
    from soundrestorer.tasks.denoise_stft import apply_mag_mask  # if provided
    x = torch.randn(1,1,8000) * 0.05
    X = stft_complex(x, n_fft=512, hop_length=128)
    M = torch.ones_like(X.abs())  # unity magnitude mask
    Y = apply_mag_mask(X, M) if callable(apply_mag_mask) else X * (M.unsqueeze(-1) if M.dim()==4 else 1.0)
    y = istft_complex(Y, n_fft=512, hop_length=128, length=x.shape[-1])
    rel = float(((x - y)**2).mean() / (x**2).mean())
    assert rel < 1e-4
