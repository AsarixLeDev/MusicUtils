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
    """
    Skip pathological batches via:
      - hard loss clip (optional)
      - NaN/Inf & RMS/peak sanity
      - SNR floor (SI-SDR proxy)
      - MAD outlier gate with floors + relative factor
    Has a 'relax()' rescue to soften thresholds dynamically.
    """
    def __init__(self,
                 hard_clip: float = 0.0,
                 window: int = 512,
                 mad_k: float = 6.0,
                 snr_floor_db: float = -6.0,
                 min_rms_db: float = -70.0,
                 max_peak: float = 1.2,
                 # robust MAD gate parameters:
                 mad_floor_abs: float = 0.20,   # absolute MAD floor (loss units)
                 mad_floor_rel: float = 0.20,   # relative MAD floor as fraction of |median|
                 rel_factor: float   = 1.6,     # also require s > median * rel_factor
                 min_hist: int       = 64):
        self.hard_clip   = float(hard_clip)
        self.window      = int(window)
        self.mad_k       = float(mad_k)
        self.snr_floor   = float(snr_floor_db)
        self.min_rms     = 10.0 ** (min_rms_db / 20.0)
        self.max_peak    = float(max_peak)
        self.mad_floor_abs = float(mad_floor_abs)
        self.mad_floor_rel = float(mad_floor_rel)
        self.rel_factor    = float(rel_factor)
        self.min_hist      = int(min_hist)

        self.hist = deque(maxlen=self.window)
        self.enable_mad = True  # can be toggled off by trainer if needed

    def relax(self, factor: float = 1.25):
        """Soften the MAD gate when too many batches get skipped."""
    #    print(f"[guard] relax: mad_k {self.mad_k:.2f}->{self.mad_k*factor:.2f}, rel_factor {self.rel_factor:.2f}->{self.rel_factor*1.05:.2f}")
        self.mad_k *= factor
        self.rel_factor *= 1.05

    def _push(self, v: float):
        if math.isfinite(v):
            self.hist.append(float(v))

    def _outlier(self, s: float) -> bool:
        if not self.enable_mad:
            return False
        if len(self.hist) < self.min_hist:
            return False
        med = statistics.median(self.hist)
        mad = statistics.median([abs(x - med) for x in self.hist]) + 1e-6
        # robust floor: absolute and relative to median
        mad_eff = max(mad, self.mad_floor_abs, self.mad_floor_rel * max(0.1, abs(med)))
        thr = med + self.mad_k * mad_eff
        # require both: above median+k*MAD and significantly above median
        return (s > thr) and (s > med * self.rel_factor)

    @torch.no_grad()
    def should_skip(
        self,
        batch,
        outputs,
        loss_value: float | None = None,
        loss_tensor: torch.Tensor | None = None,
    ) -> Tuple[bool, Optional[str]]:
        # ---- loss scalar for gates (prefer tensor path; cheap CPU copy) ----
        s_mad: float | None = None
        if loss_tensor is not None and torch.is_tensor(loss_tensor):
            if self.hard_clip > 0:
                try:
                    if bool((loss_tensor > self.hard_clip).detach().to("cpu").item()):
                        lv = float(loss_tensor.detach().to("cpu"))
                        return True, f"loss {lv:.3f} > hard_clip {self.hard_clip:.3f}"
                except Exception:
                    pass
            try:
                s_mad = float(loss_tensor.detach().to("cpu"))
            except Exception:
                s_mad = None
        elif loss_value is not None:
            s_mad = float(loss_value)
            if self.hard_clip > 0 and s_mad > self.hard_clip:
                return True, f"loss {s_mad:.3f} > hard_clip {self.hard_clip:.3f}"

        # ---- input/output sanity ----
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]
            for nm, t in (("noisy", noisy), ("clean", clean)):
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    return True, f"{nm} has NaN/Inf"
            yh = outputs.get("yhat") if isinstance(outputs, dict) else None
            if isinstance(yh, torch.Tensor) and not torch.isfinite(yh).all():
                return True, "yhat has NaN/Inf"

            nm = _bt(noisy); cm = _bt(clean)
            rms_n = torch.sqrt(torch.clamp((nm**2).mean(dim=-1), 1e-12)).mean()
            rms_c = torch.sqrt(torch.clamp((cm**2).mean(dim=-1), 1e-12)).mean()
            if float(rms_n) < self.min_rms or float(rms_c) < self.min_rms:
                return True, f"too silent (rms n={float(rms_n):.2e}, c={float(rms_c):.2e})"
            peak_n = float(torch.max(torch.abs(nm))); peak_c = float(torch.max(torch.abs(cm)))
            if peak_n > self.max_peak or peak_c > self.max_peak:
                return True, f"peak too high (n={peak_n:.2f}, c={peak_c:.2f})"

            try:
                snr_now = float(si_sdr_db(noisy, clean).mean())
                if snr_now < self.snr_floor:
                    return True, f"snr {snr_now:.2f} < floor {self.snr_floor:.2f}"
            except Exception:
                pass

        # ---- MAD outlier (dynamic) ----
        if s_mad is not None and self._outlier(s_mad):
            med = statistics.median(self.hist) if self.hist else 0.0
            return True, f"MAD outlier {s_mad:.3f} (med~{med:.3f})"

        if s_mad is not None:
            self._push(s_mad)
        return False, None
