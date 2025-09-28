import torch
from .base import LossModule
from ..core.registry import LOSSES
from ..utils.audio import mel_filterbank, stft_any, stft_phase_cosine, si_sdr_db

@LOSSES.register("l1_wave")
class L1Wave(LossModule):
    def forward(self, outputs, batch):
        return torch.mean(torch.abs(outputs["yhat"] - outputs["clean"].float()))

@LOSSES.register("sisdr_ratio")
class SISDRRatio(LossModule):
    def __init__(self, min_db=0.0, cap=1.0):
        self.min_db = float(min_db); self.cap = float(cap)
    def forward(self, outputs, batch):
        eps = 1e-8
        y = outputs["yhat"]; x = outputs["clean"].float()
        if y.dim()==3: y=y.mean(dim=1); x=x.mean(dim=1)
        xz = x - x.mean(dim=-1, keepdim=True)
        yz = y - y.mean(dim=-1, keepdim=True)
        s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
             (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
        e = yz - s
        si_sdr = 10 * torch.log10(torch.sum(s**2, dim=-1) /
                                  (torch.sum(e**2, dim=-1) + eps) + eps)
        loss = torch.clamp((self.min_db - si_sdr) / max(abs(self.min_db), 1e-6), min=0.0)
        if self.cap > 0: loss = torch.clamp(loss, max=self.cap)
        return loss.mean()

@LOSSES.register("mel_l1")
class MelL1(LossModule):
    def __init__(self, sr=48000, n_mels=64, n_fft=1024, hop=256, eps=1e-4):
        self.sr=int(sr); self.n_mels=int(n_mels); self.n_fft=int(n_fft); self.hop=int(hop); self.eps=float(eps)
        self._fb=None; self._dev=None
    def _fb_on(self, device):
        if (self._fb is None) or (self._dev != device):
            self._fb = mel_filterbank(self.sr, self.n_fft, self.n_mels, device=device)
            self._dev = device
        return self._fb
    def forward(self, outputs, batch):
        from ..utils.audio import hann_window
        y = outputs["yhat"]; x = outputs["clean"].float()
        win = hann_window(self.n_fft, y.device)
        Ym = torch.matmul(self._fb_on(y.device), stft_any(y, self.n_fft, self.hop, win).abs())
        Xm = torch.matmul(self._fb_on(x.device), stft_any(x, self.n_fft, self.hop, win).abs())
        return torch.mean(torch.abs(torch.log10(Ym + self.eps) - torch.log10(Xm + self.eps)))

@LOSSES.register("highband_l1")
class HighbandL1(LossModule):
    def __init__(self, sr=48000, cutoff_khz=8.0, n_fft=1024, hop=256, use_log=True, eps=1e-6):
        self.sr=int(sr); self.cut_khz=float(cutoff_khz); self.n_fft=int(n_fft); self.hop=int(hop)
        self.use_log=bool(use_log); self.eps=float(eps)
    def forward(self, outputs, batch):
        from ..utils.audio import hann_window, stft_any
        y = outputs["yhat"]; x = outputs["clean"].float()
        win = hann_window(self.n_fft, y.device)
        Y = stft_any(y, self.n_fft, self.hop, win).abs()
        X = stft_any(x, self.n_fft, self.hop, win).abs()
        F = self.n_fft // 2 + 1
        freqs = torch.arange(F, device=y.device, dtype=y.dtype) * (float(self.sr)/float(self.n_fft))
        mask = freqs >= (self.cut_khz*1000.0)
        Yh, Xh = Y[:, mask, :], X[:, mask, :]
        diff = (torch.abs(torch.log1p(Yh + self.eps) - torch.log1p(Xh + self.eps)) if self.use_log
                else torch.abs(Yh - Xh))
        return diff.mean()

@LOSSES.register("phase_cosine")
class PhaseCosine(LossModule):
    def __init__(self, n_fft=1024, hop=256, mag_weight=True):
        self.n_fft=int(n_fft); self.hop=int(hop); self.mag_weight=bool(mag_weight)
    def forward(self, outputs, batch):
        from ..utils.audio import hann_window
        return stft_phase_cosine(outputs["yhat"], outputs["clean"].float(),
                                 hann_window(self.n_fft, outputs["yhat"].device),
                                 n_fft=self.n_fft, hop=self.hop,
                                 mag_weight=self.mag_weight)

@LOSSES.register("energy_anchor")
class EnergyAnchor(LossModule):
    def __init__(self, strength=0.005):
        self.strength=float(strength)
    def forward(self, outputs, batch):
        y = outputs["yhat"].float(); x = outputs["noisy"].float()
        y_std = y.std(dim=-1).mean(); x_std = x.std(dim=-1).mean()
        return self.strength * ((y_std / (x_std + 1e-8)) - 1.0).pow(2)
