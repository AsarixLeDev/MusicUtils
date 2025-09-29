import torch
from .base import LossModule
from ..core.registry import LOSSES

def _stft_any(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: x = x.unsqueeze(0)
    elif x.dim() == 3: x = x.mean(dim=1)
    elif x.dim() != 2: raise RuntimeError(f"Unexpected x.dim()={x.dim()}")
    # try to keep dtype with window; fallback to float32 if needed
    want_dtype = win.dtype
    if x.dtype != want_dtype:
        x = x.to(want_dtype)
    try:
        return torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                          center=True, return_complex=True)
    except Exception:
        x32 = x.to(torch.float32); w32 = win.to(torch.float32)
        return torch.stft(x32, n_fft=n_fft, hop_length=hop, window=w32,
                          center=True, return_complex=True)

def _spectral_convergence_vec(y_mag: torch.Tensor, x_mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample SC: returns a (B,) tensor. We’ll mask near-silence at the caller.
    """
    B = y_mag.shape[0]
    yv = y_mag.reshape(B, -1).to(torch.float32)
    xv = x_mag.reshape(B, -1).to(torch.float32)
    num = torch.linalg.vector_norm(yv - xv, ord=2, dim=1)
    den = torch.linalg.vector_norm(xv,      ord=2, dim=1).clamp_min(eps)
    return num / den  # (B,)

def _log_mag_l1(y_mag: torch.Tensor, x_mag: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = torch.log10(y_mag.to(torch.float32) + eps)
    x = torch.log10(x_mag.to(torch.float32) + eps)
    return torch.mean(torch.abs(y - x))

@LOSSES.register("mrstft")
class MultiResSTFTLoss(LossModule):
    def __init__(self, fft_sizes=(1024,512,2048), hops=(256,128,512), win_lengths=(1024,512,2048),
                 sc_weight=0.5, mag_weight=0.5,
                 sc_ignore_silence: bool = True, silence_rms: float = 1e-3):
        self.fft_sizes   = list(fft_sizes)
        self.hops        = list(hops)
        self.win_lengths = list(win_lengths)
        self.sc_weight   = float(sc_weight)
        self.mag_weight  = float(mag_weight)
        self.sc_ignore_silence = bool(sc_ignore_silence)
        self.silence_rms = float(silence_rms)
        self._wins = {}

    def _win(self, n_fft: int, device, dtype):
        key = (n_fft, device, dtype)
        w = self._wins.get(key, None)
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.hann_window(n_fft, device=device, dtype=dtype)
            self._wins[key] = w
        return w

    def forward(self, outputs, batch):
        y = outputs["yhat"]
        x = outputs["clean"]
        device = y.device
        stft_dtype = y.dtype if y.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

        # per-sample RMS to detect silence (time domain, stable)
        xm = x if x.dim() == 2 else x.mean(dim=1)
        rms = xm.to(torch.float32).pow(2).mean(dim=-1).sqrt()  # (B,)
        if self.sc_ignore_silence:
            w_sc = (rms > self.silence_rms).to(torch.float32)  # (B,)
        else:
            w_sc = torch.ones_like(rms, dtype=torch.float32)

        sc_acc = 0.0
        mag_acc = 0.0
        nsc = 0
        nmag = 0

        for n_fft, hop, wl in zip(self.fft_sizes, self.hops, self.win_lengths):
            win = self._win(n_fft, device, stft_dtype)
            Y = _stft_any(y, n_fft, hop, win); X = _stft_any(x, n_fft, hop, win)
            Ymag = Y.abs(); Xmag = X.abs()

            # SC (masked average across batch)
            sc_vec = _spectral_convergence_vec(Ymag, Xmag)  # (B,)
            denom = w_sc.sum().clamp_min(1.0)
            sc_acc += (sc_vec * w_sc).sum() / denom
            nsc += 1

            # log-mag L1 (simple batch mean)
            mag_acc += _log_mag_l1(Ymag, Xmag)
            nmag += 1

        sc_avg  = sc_acc / max(1, nsc)
        mag_avg = mag_acc / max(1, nmag)
        return self.sc_weight * sc_avg + self.mag_weight * mag_avg
