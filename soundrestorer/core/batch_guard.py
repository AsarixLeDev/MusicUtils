from __future__ import annotations
import torch, math, statistics
from collections import deque
from typing import Tuple, Optional

def _bt(x: torch.Tensor) -> torch.Tensor:
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
    return 10.0 * torch.log10(num / den + eps)

class BatchGuard:
    def __init__(self, hard_clip=0.0, window=512, mad_k=6.0,
                 snr_floor_db=-6.0, min_rms_db=-70.0, max_peak=1.2):
        self.hard_clip = float(hard_clip)      # 0 disables static clip
        self.window = int(window)
        self.mad_k = float(mad_k)
        self.snr_floor = float(snr_floor_db)
        self.min_rms = 10.0 ** (min_rms_db / 20.0)
        self.max_peak = float(max_peak)
        self.hist = deque(maxlen=self.window)

    def _push(self, v):
        if math.isfinite(v): self.hist.append(float(v))

    def _outlier(self, s: float) -> bool:
        if len(self.hist) < 16: return False
        med = statistics.median(self.hist)
        mad = statistics.median([abs(x - med) for x in self.hist]) + 1e-6
        return s > (med + self.mad_k * mad)

    @torch.no_grad()
    def should_skip(self, batch, outputs, loss_value: float) -> Tuple[bool, Optional[str]]:
        s = float(loss_value)
        if self.hard_clip > 0 and s > self.hard_clip:
            return True, f"loss {s:.3f} > hard_clip {self.hard_clip:.3f}"

        # Expect (noisy, clean, ...)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]
            for nm, t in (("noisy", noisy), ("clean", clean)):
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    return True, f"{nm} has NaN/Inf"
            yh = outputs.get("yhat") if isinstance(outputs, dict) else None
            if isinstance(yh, torch.Tensor) and not torch.isfinite(yh).all():
                return True, "yhat has NaN/Inf"

            nm = _bt(noisy); cm = _bt(clean)
            rms_n = torch.sqrt(torch.clamp((nm**2).mean(dim=-1), 1e-12)).mean().item()
            rms_c = torch.sqrt(torch.clamp((cm**2).mean(dim=-1), 1e-12)).mean().item()
            if rms_n < self.min_rms or rms_c < self.min_rms:
                return True, f"too silent (rms n={rms_n:.2e}, c={rms_c:.2e})"
            peak_n = float(torch.max(torch.abs(nm))); peak_c = float(torch.max(torch.abs(cm)))
            if peak_n > self.max_peak or peak_c > self.max_peak:
                return True, f"peak too high (n={peak_n:.2f}, c={peak_c:.2f})"

            try:
                snr_now = si_sdr_db(noisy, clean).mean().item()
                if snr_now < self.snr_floor:
                    return True, f"snr {snr_now:.2f} < floor {self.snr_floor:.2f}"
            except Exception:
                pass

        if self._outlier(s):
            med = statistics.median(self.hist)
            return True, f"MAD outlier {s:.3f} (med~{med:.3f})"

        self._push(s)
        return False, None
