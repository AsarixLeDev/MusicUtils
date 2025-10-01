import torch

from .callbacks import Callback
from ..utils.noise import NoiseFactory, mix_at_snr

class EnsureMinSNRCallback(Callback):
    """
    Per-item rescue: if SNR(noisy, clean) is above min_snr_db (too clean),
    inject procedural noise to bring it into a target SNR window.
    Train-only by default; does *not* touch validation unless you set train_only=False.
    """

    def __init__(self, sr: int, min_snr_db: float = 25.0,
                 snr_min: float = 4.0, snr_max: float = 20.0,
                 out_peak: float = 0.98, train_only: bool = True):
        self.sr = int(sr)
        self.min_snr_db = float(min_snr_db)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.out_peak = float(out_peak)
        self.train_only = bool(train_only)
        # we reuse your NoiseFactory
        self.noise = NoiseFactory(self.sr, {})
        # stats
        self._seen = 0
        self._fixed = 0

    def on_epoch_start(self, **_):
        self._seen = 0
        self._fixed = 0

    @torch.no_grad()
    def on_batch_start(self, trainer=None, state=None, batch=None, **_):
        if self.train_only and trainer and trainer.model.training is False:
            return
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            return
        noisy, clean = batch[0], batch[1]

        # to (B,T) mono float32 on-device
        def _bt(t):
            if t.dim() == 3:
                t = t.mean(dim=1)
            elif t.dim() == 1:
                t = t.unsqueeze(0)
            return t

        noisy_bt = _bt(noisy).to(torch.float32)
        clean_bt = _bt(clean).to(torch.float32)

        B, T = noisy_bt.shape
        self._seen += B

        # SNR(noisy vs clean) ~ 10*log10(||clean||^2 / ||noisy-clean||^2)
        resid = noisy_bt - clean_bt
        num = (clean_bt ** 2).sum(dim=-1).clamp_min(1e-12)
        den = (resid ** 2).sum(dim=-1).clamp_min(1e-12)
        snr_db = 10.0 * torch.log10(num / den)

        # Items to fix
        mask = snr_db > self.min_snr_db
        if not mask.any():
            return

        # Build per-item target SNRs in [snr_min, snr_max]
        tgt = torch.empty(B, device=clean.device, dtype=torch.float32).uniform_(self.snr_min, self.snr_max)
        tgt = tgt[mask]

        # Sample procedural noise and mix
        noises = self.noise.sample_batch(int(mask.sum().item()), T, device=clean.device)  # (K,T)
        # mix_at_snr expects (B,T) clean/noise and target SNR per-item
        new_noisy = mix_at_snr(clean_bt[mask], noises, tgt, peak=self.out_peak)  # (K,T)

        # write back in place respecting original shape
        if noisy.dim() == 2:
            noisy[mask] = new_noisy
        elif noisy.dim() == 3:
            # (B,C,T): broadcast new_noisy (B,T) -> (B,1,T) then expand
            noisy[mask] = new_noisy.unsqueeze(1).expand(-1, noisy.shape[1], -1)
        else:
            raise RuntimeError(f"Unexpected noisy shape {tuple(noisy.shape)}")

        self._fixed += int(mask.sum().item())

    def on_epoch_end(self, trainer=None, state=None, **_):
        if self._seen > 0:
            frac = self._fixed / float(self._seen)
            print(f"[min-snr] epoch {state.epoch}: fixed {self._fixed}/{self._seen} items "
                  f"({frac * 100:.1f}%) with SNR>{self.min_snr_db} dB")
            try:
                trainer.state.info["min_snr_fixed_frac"] = frac
            except Exception:
                pass
