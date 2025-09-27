# soundrestorer/core/batch_guard.py
from __future__ import annotations
import torch, math, statistics
from collections import deque
from typing import Tuple, Optional

def _bt(x: torch.Tensor) -> torch.Tensor:
    # (B,T) ok; (B,1,T) or (B,C,T) -> mono (B,T); (T,) -> (1,T)
    if x.dim() == 1: return x.unsqueeze(0)
    if x.dim() == 2: return x
    if x.dim() == 3: return x.mean(dim=1)
    raise RuntimeError(f"expected 1/2/3D audio, got {tuple(x.shape)}")

@torch.no_grad()
def si_sdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = _bt(y); x = _bt(x)
    xz = x - x.mean(dim=-1, keepdim=True)
    yz = y - y.mean(dim=-1, keepdim=True)
    s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
         (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
    e = yz - s
    num = torch.sum(s**2, dim=-1)
    den = torch.sum(e**2, dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)  # (B,)

class BatchGuard:
    """
    Skips pathological batches using several guards:
      - NaN/Inf in inputs or outputs
      - input RMS too low/high, peak too high
      - SNR gate (SI-SDR(noisy,clean) below a floor)
      - dynamic outlier: loss > median + k*MAD over a rolling window
      - hard clip: loss > hard_clip
    """
    def __init__(self,
                 hard_clip: float = 2.0,
                 window: int = 512,
                 mad_k: float = 6.0,
                 snr_floor_db: float = -3.0,   # allow slightly negative if you train with hard SNR
                 min_rms_db: float = -70.0,
                 max_peak: float = 1.2):
        self.hard_clip = float(hard_clip)
        self.window = int(window)
        self.mad_k = float(mad_k)
        self.snr_floor_db = float(snr_floor_db)
        self.min_rms = 10.0 ** (min_rms_db / 20.0)
        self.max_peak = float(max_peak)
        self.loss_hist = deque(maxlen=self.window)

    def update_hist(self, loss_val: float):
        if math.isfinite(loss_val):
            self.loss_hist.append(float(loss_val))

    def _dyn_outlier(self, s: float) -> bool:
        if len(self.loss_hist) < 16:
            return False
        median = statistics.median(self.loss_hist)
        devs = [abs(v - median) for v in self.loss_hist]
        mad = statistics.median(devs) + 1e-6
        return s > (median + self.mad_k * mad)

    @torch.no_grad()
    def should_skip(self, batch, outputs, loss_value: float) -> Tuple[bool, Optional[str]]:
        s = float(loss_value)

        # 0) hard clip
        if self.hard_clip > 0 and s > self.hard_clip:
            return True, f"loss {s:.3f} > hard_clip {self.hard_clip:.3f}"

        # Expect batch as (noisy, clean, ...)
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            self.update_hist(s)
            return False, None

        noisy, clean = batch[0], batch[1]

        # 1) NaN/Inf in inputs or outputs
        for name, t in (("noisy", noisy), ("clean", clean)):
            if isinstance(t, torch.Tensor):
                if not torch.isfinite(t).all():
                    return True, f"{name} contains NaN/Inf"

        yh = outputs.get("yhat") if isinstance(outputs, dict) else None
        if isinstance(yh, torch.Tensor) and not torch.isfinite(yh).all():
            return True, "yhat contains NaN/Inf"

        # 2) RMS/peak sanity on inputs
        nm = _bt(noisy).detach()
        cm = _bt(clean).detach()
        rms_n = torch.sqrt(torch.clamp((nm**2).mean(dim=-1), min=1e-12)).mean().item()
        rms_c = torch.sqrt(torch.clamp((cm**2).mean(dim=-1), min=1e-12)).mean().item()
        if rms_c < self.min_rms or rms_n < self.min_rms:
            return True, f"too silent (rms c={rms_c:.2e}, n={rms_n:.2e})"
        peak_n = torch.max(torch.abs(nm)).item()
        peak_c = torch.max(torch.abs(cm)).item()
        if peak_n > self.max_peak or peak_c > self.max_peak:
            return True, f"peak too high (n={peak_n:.2f}, c={peak_c:.2f})"

        # 3) SNR gate (fast SI-SDR proxy)
        try:
            snr_now = si_sdr_db(noisy, clean).mean().item()
            if snr_now < self.snr_floor_db:
                return True, f"snr floor {snr_now:.2f} < {self.snr_floor_db:.2f}"
        except Exception:
            pass

        # 4) dynamic outlier vs rolling distribution
        if self._dyn_outlier(s):
            med = statistics.median(self.loss_hist)
            return True, f"dyn outlier loss {s:.3f} > med+{self.mad_k}*MAD ~ {med:.3f}"

        # ok — record and keep
        self.update_hist(s)
        return False, None
