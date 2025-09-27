# Keep your MR-STFT implementation, wrapped as a LossModule.
import torch, torch.nn as nn
from .base import LossModule
from ..core.registry import LOSSES

def _stft_any(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: x = x.unsqueeze(0)
    elif x.dim() == 3: x = x.mean(dim=1)
    elif x.dim() != 2: raise RuntimeError
    return torch.stft(x.float(), n_fft=n_fft, hop_length=hop, window=win,
                      center=True, return_complex=True)

def _spectral_convergence(y_mag: torch.Tensor, x_mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B = y_mag.shape[0]
    yv = y_mag.reshape(B, -1).float()
    xv = x_mag.reshape(B, -1).float()
    num = torch.linalg.vector_norm(yv - xv, ord=2, dim=1)
    den = torch.linalg.vector_norm(xv,      ord=2, dim=1).clamp_min(eps)
    return (num / den).mean()

def _log_mag_l1(y_mag: torch.Tensor, x_mag: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.mean(torch.abs(torch.log10(y_mag.float() + eps) - torch.log10(x_mag.float() + eps)))

@LOSSES.register("mrstft")
class MultiResSTFTLoss(LossModule):
    def __init__(self, fft_sizes=(1024,512,2048), hops=(256,128,512), win_lengths=(1024,512,2048),
                 sc_weight=0.5, mag_weight=0.5):
        self.fft_sizes = list(fft_sizes)
        self.hops = list(hops)
        self.win_lengths = list(win_lengths)
        self.sc_weight = float(sc_weight); self.mag_weight = float(mag_weight)
        self._wins = {}

    def _win(self, n_fft: int, device):
        key = (n_fft, device)
        if key not in self._wins:
            self._wins[key] = torch.hann_window(n_fft, device=device, dtype=torch.float32)
        return self._wins[key]

    def forward(self, outputs, batch):
        y = outputs["yhat"]; x = outputs["clean"]
        device = y.device
        sc_tot = 0.0; mag_tot = 0.0; n = len(self.fft_sizes)
        for n_fft, hop, wl in zip(self.fft_sizes, self.hops, self.win_lengths):
            win = self._win(n_fft, device)
            Y = _stft_any(y, n_fft, hop, win); X = _stft_any(x, n_fft, hop, win)
            sc = _spectral_convergence(Y.abs(), X.abs()); mag = _log_mag_l1(Y.abs(), X.abs())
            sc_tot += sc; mag_tot += mag
        sc_avg = sc_tot / n; mag_avg = mag_tot / n
        return self.sc_weight * sc_avg + self.mag_weight * mag_avg
