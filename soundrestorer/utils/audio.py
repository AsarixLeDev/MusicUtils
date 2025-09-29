from __future__ import annotations
import torch

def hann_window(n_fft, device, dtype=torch.float32):
    # sqrt-Hann for COLA
    w = torch.hann_window(n_fft, periodic=True, device=device, dtype=dtype if dtype is not None else torch.float32)
    return torch.sqrt(torch.clamp(w, min=0)).to(dtype or torch.float32)

def _as_batch_mono(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3: return x.mean(dim=1)
    raise RuntimeError(f"Expected 1D/2D/3D waveform, got shape {tuple(x.shape)}")


@torch.no_grad()
def sisdr_improvement_db(
    yhat: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    eps: float = 1e-8,
    trim_frac: float = 0.0,
) -> torch.Tensor:
    """
    Mean (or trimmed mean) SI-SDR improvement in dB across the batch.
    """
    si_n = si_sdr_db(noisy, clean, eps=eps)      # (B,)
    si_r = si_sdr_db(yhat,  clean, eps=eps)      # (B,)
    d = si_r - si_n                               # (B,)

    if trim_frac <= 0:
        return d.mean()

    k = int(d.numel() * trim_frac)
    if k <= 0:
        return d.mean()
    d_sorted, _ = torch.sort(d)
    core = d_sorted[k: d_sorted.numel() - k]
    if core.numel() == 0:
        core = d_sorted
    return core.mean()

@torch.no_grad()
def si_sdr_db(
    y: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
    clamp_db_min: float | None = -60.0,
    match_length: bool = True,
    debug_shapes: int = 0,         # <— NEW: log shapes for first N calls
) -> torch.Tensor:
    """
    Robust scale-invariant SDR in dB, with optional one-shot shape logging.

    Accepts (B,T) or (B,C,T) (and (T)/(C,T), auto-batched).
    If either has channels, BOTH are downmixed to mono to avoid broadcasting.
    If match_length=True, trims both to min length.
    """
    # -------- tiny helper to print once/few times --------
    # Reduce noise: only print for the first `debug_shapes` invocations (per process)
    if not hasattr(si_sdr_db, "_dbg_count"):
        si_sdr_db._dbg_count = 0

    def _pfx(ok: bool) -> str:
        return "[si_sdr_db]" if ok else "[si_sdr_db:WARN]"

    def _stats(name: str, t: torch.Tensor) -> str:
        t32 = t.detach()
        m = float(t32.mean()) if t32.numel() else 0.0
        s = float(t32.std())  if t32.numel() else 0.0
        fin = bool(torch.isfinite(t32).all())
        return f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, dev={t.device}, mean={m:.3e}, std={s:.3e}, finite={fin}"

    log_now = si_sdr_db._dbg_count < int(debug_shapes)

    # --- normalize shapes to (B,T) ---
    def _to_bt(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:  return t.unsqueeze(0)
        if t.dim() == 2:  return t
        if t.dim() == 3:  return t.mean(dim=1)
        raise RuntimeError(f"Unexpected tensor shape {tuple(t.shape)} for SI-SDR")

    y0, x0 = y, x  # keep originals for debug prints
    y = _to_bt(y); x = _to_bt(x)

    # --- optional length alignment ---
    if match_length:
        T = min(y.shape[-1], x.shape[-1])
        if y.shape[-1] != T: y = y[..., :T]
        if x.shape[-1] != T: x = x[..., :T]

    if zero_mean:
        y = y - y.mean(dim=-1, keepdim=True)
        x = x - x.mean(dim=-1, keepdim=True)

    if log_now:
        print(_pfx(True), "INPUTS")
        print(" ", _stats("y_in", y0))
        print(" ", _stats("x_in", x0))
        print(_pfx(True), "NORMALIZED")
        print(" ", _stats("y_bt", y))
        print(" ", _stats("x_bt", x))
        si_sdr_db._dbg_count += 1

    # basic sanity checks
    assert y.shape[:2] == x.shape[:2], f"Batch/length mismatch after normalization: y{y.shape} vs x{x.shape}"

    # --- projection ---
    xx = torch.sum(x * x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.sum(y * x, dim=-1, keepdim=True) / xx
    s = scale * x
    e = y - s

    num = torch.sum(s * s, dim=-1).clamp_min(eps)
    den = torch.sum(e * e, dim=-1).clamp_min(eps)
    sisdr = 10.0 * torch.log10(num / den)

    if clamp_db_min is not None:
        floor = torch.tensor(clamp_db_min, device=sisdr.device, dtype=sisdr.dtype)
        sisdr = torch.maximum(sisdr, floor)

    bad = ~torch.isfinite(sisdr)
    if bad.any():
        fill = torch.tensor(clamp_db_min if clamp_db_min is not None else -60.0,
                            device=sisdr.device, dtype=sisdr.dtype)
        sisdr = sisdr.masked_fill(bad, fill)

    if log_now:
        print(_pfx(True), f"SI-SDR batch={tuple(sisdr.shape)} mean={float(sisdr.mean()):.2f} dB (min={float(sisdr.min()):.2f}, max={float(sisdr.max()):.2f})")

    return sisdr



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
