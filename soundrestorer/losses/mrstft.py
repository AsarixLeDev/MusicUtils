from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["MultiResSTFTLoss"]

def _stft_any(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    """
    x: (B,T) or (B,C,T) or (T) -> complex STFT (B,F,T)
    Always mixes multi-channel to mono internally for stability.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1,T)
    elif x.dim() == 3:
        x = x.mean(dim=1)   # (B,C,T) -> (B,T)
    elif x.dim() != 2:
        raise RuntimeError(f"STFT expects 1D/2D/3D, got shape {tuple(x.shape)}")

    return torch.stft(
        x.float(), n_fft=n_fft, hop_length=hop, window=win,
        center=True, return_complex=True
    )  # (B,F,T)

def _spectral_convergence(y_mag: torch.Tensor, x_mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SC per batch item: ||Y-X||_2 / ||X||_2 over flattened (F*T), then mean over batch.
    """
    B = y_mag.shape[0]
    yv = y_mag.reshape(B, -1).float()
    xv = x_mag.reshape(B, -1).float()
    num = torch.linalg.vector_norm(yv - xv, ord=2, dim=1)
    den = torch.linalg.vector_norm(xv,      ord=2, dim=1).clamp_min(eps)
    return (num / den).mean()

def _log_mag_l1(y_mag: torch.Tensor, x_mag: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.mean(torch.abs(torch.log10(y_mag.float() + eps) - torch.log10(x_mag.float() + eps)))

class MultiResSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.
    Accepts y,x shaped (B,T) or (B,C,T). Returns (total, sc, mag) scalars.
    """
    def __init__(self,
                 fft_sizes=(1024, 512, 2048),
                 hops=(256, 128, 512),
                 win_lengths=(1024, 512, 2048),
                 sc_weight: float = 0.5,
                 mag_weight: float = 0.5):
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(win_lengths)
        self.fft_sizes = list(fft_sizes)
        self.hops = list(hops)
        self.win_lengths = list(win_lengths)
        self.sc_weight = float(sc_weight)
        self.mag_weight = float(mag_weight)
        self._windows = {}  # (n_fft, device) -> window

    def _get_win(self, n_fft: int, device) -> torch.Tensor:
        key = (n_fft, device)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(n_fft, device=device, dtype=torch.float32)
        return self._windows[key]

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (B,C,T) is accepted — we collapse to mono inside _stft_any.
        device = y.device
        sc_total = 0.0
        mag_total = 0.0
        n = len(self.fft_sizes)

        for n_fft, hop, win_len in zip(self.fft_sizes, self.hops, self.win_lengths):
            win = self._get_win(n_fft, device)
            Y = _stft_any(y, n_fft, hop, win)   # (B,F,T)
            X = _stft_any(x, n_fft, hop, win)
            Ymag, Xmag = Y.abs(), X.abs()

            sc  = _spectral_convergence(Ymag, Xmag)
            mag = _log_mag_l1(Ymag, Xmag)

            sc_total  = sc_total  + sc
            mag_total = mag_total + mag

        sc_avg  = sc_total  / n
        mag_avg = mag_total / n
        total   = self.sc_weight * sc_avg + self.mag_weight * mag_avg
        return total, sc_avg, mag_avg
