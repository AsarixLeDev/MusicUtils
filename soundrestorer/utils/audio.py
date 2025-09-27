from __future__ import annotations
import torch

def hann_window(n_fft, device):
    # sqrt-Hann for COLA
    w = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)
    return torch.sqrt(torch.clamp(w, min=0))

def _as_batch_mono(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3: return x.mean(dim=1)
    raise RuntimeError(f"Expected 1D/2D/3D waveform, got shape {tuple(x.shape)}")

@torch.no_grad()
def si_sdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if y.dim() == 3: y = y.mean(dim=1); x = x.mean(dim=1)
    xz = x - x.mean(dim=-1, keepdim=True)
    yz = y - y.mean(dim=-1, keepdim=True)
    s = (torch.sum(yz * xz, dim=-1, keepdim=True) / (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
    e = yz - s
    num = torch.sum(s ** 2, dim=-1)
    den = torch.sum(e ** 2, dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)

def stft_any(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    x = _as_batch_mono(x)
    win = win.contiguous().to(dtype=torch.float32, device=x.device)
    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=False):
        X = torch.stft(x.float(), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, center=True, return_complex=True, normalized=False)
    return X  # (B,F,T) complex

def stft_pair(wave: torch.Tensor, win: torch.Tensor, n_fft: int, hop: int):
    X = stft_any(wave, n_fft, hop, win)
    Xri = torch.stack([X.real, X.imag], dim=1).contiguous()  # (B,2,F,T)
    return X, Xri

def istft_from(X: torch.Tensor, win: torch.Tensor, length: int, n_fft: int, hop: int):
    win = win.contiguous().to(dtype=torch.float32, device=X.device)
    device_type = "cuda" if X.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=False):
        X = X.to(torch.complex64)
        y = torch.istft(X, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, center=True, length=length, normalized=False)
    return y  # (B,T)

# ---- extra pieces used by losses ----
def _hz_to_mel(f):  # HTK
    return 2595.0 * torch.log10(1.0 + f / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None, device=None):
    fmax = float(sr) / 2.0 if fmax is None else float(fmax)
    m_min, m_max = _hz_to_mel(torch.tensor(fmin)), _hz_to_mel(torch.tensor(fmax))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    bins = torch.floor((n_fft + 1) * f_pts / float(sr)).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left: center] = torch.linspace(0, 1, center - left)
        if right > center:
            fb[i, center: right] = torch.linspace(1, 0, right - center)
    return fb.to(device=device, dtype=torch.float32)

def stft_phase_cosine(y: torch.Tensor, x: torch.Tensor, win: torch.Tensor, n_fft: int, hop: int,
                      eps: float = 1e-8, mag_weight: bool = True) -> torch.Tensor:
    Y = stft_any(y, n_fft, hop, win)
    X = stft_any(x, n_fft, hop, win)
    Yr, Yi = Y.real, Y.imag; Xr, Xi = X.real, X.imag
    dot = Yr * Xr + Yi * Xi
    denom = (Y.abs() * X.abs() + eps)
    cos_dphi = dot / denom
    if mag_weight:
        w = X.abs()
        return (w * (1.0 - cos_dphi)).sum() / (w.sum() + eps)
    return (1.0 - cos_dphi).mean()
