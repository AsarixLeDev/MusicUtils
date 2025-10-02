from soundrestorer.callbacks.callbacks import Callback
from ..utils import snr_db
from ..utils.noise import NoiseFactory, mix_at_snr

import torch


class ProcNoiseAugmentCallback(Callback):
    """
    Replaces batch noisy = clean + procedural_noise at random, on-the-fly.
    Works without modifying your dataset. Train-only by default.
    Now tracks how many items were replaced to report the true fraction.
    """

    def __init__(self, sr, prob=0.5, snr_min=0.0, snr_max=20.0, out_peak=0.98,
                 train_only=True, noise_cfg=None, track_stats: bool = True,
                 fixed_seed: int | None = None, fixed_per_epoch: bool = False, require_replace: bool = False, ):
        self.sr = int(sr)
        self.prob = float(prob)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.out_peak = float(out_peak)
        self.train_only = bool(train_only)
        self.noise = NoiseFactory(self.sr, noise_cfg or {})
        self.track = bool(track_stats)
        self._seen = 0
        self._repl = 0
        self.fixed_seed = fixed_seed
        self.fixed_per_epoch = bool(fixed_per_epoch)
        self._fixed_noises = None  # type: Optional[torch.Tensor]
        self.require_replace = bool(require_replace)

    def on_epoch_start(self, **_):
        self._seen = 0;
        self._repl = 0
        if self.fixed_per_epoch:
            self._fixed_noises = None  # regenerate per epoch

    @torch.no_grad()
    def on_batch_start(self, trainer=None, state=None, batch=None, **_):
        # respect train_only
        if self.train_only and trainer and getattr(trainer, "model", None) and trainer.model.training is False:
            return

        # ---- accept dict or tuple/list ----
        noisy = clean = None
        if isinstance(batch, dict):
            noisy = batch.get("noisy", None)
            clean = batch.get("clean", None)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]
        else:
            return
        if noisy is None or clean is None:
            return

        device = clean.device

        # normalize shapes to (B,T) for SNR math; keep original for write-back
        def _bt(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:  # (B,C,T) -> (B,T) mono
                return x.mean(dim=1)
            if x.dim() == 2:  # (B,T)
                return x
            if x.dim() == 1:  # (T,) -> (1,T)
                return x.unsqueeze(0)
            raise RuntimeError(f"Unexpected tensor shape {tuple(x.shape)}")

        noisy_bt = _bt(noisy).to(torch.float32)  # keep SNR math in float32
        clean_bt = _bt(clean).to(torch.float32)
        B, T = clean_bt.shape
        if self.track:
            self._seen += B

        # ---- one-time pre-SNR telemetry (first batch seen) ----
        try:
            if self.track and self._seen <= B and B > 0:
                n = min(noisy_bt.shape[-1], clean_bt.shape[-1])
                num = (clean_bt[..., :n] ** 2).sum(dim=-1).clamp_min(1e-12)
                den = ((noisy_bt[..., :n] - clean_bt[..., :n]) ** 2).sum(dim=-1).clamp_min(1e-12)
                pre_snr = float((10.0 * torch.log10(num / den)).mean().item())
                print(f"[proc-noise] pre-mix SNR ~ {pre_snr:+.2f} dB")
        except Exception:
            pass

        # probability gate
        import random
        if random.random() > self.prob:
            return

        # ---- mix procedural noise at target SNR per item ----
        # match dtype to clean_bt (useful for future bf16/amp paths)
        noise_dtype = clean_bt.dtype
        if self.fixed_seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(self.fixed_seed)
            noises = torch.randn(B, T, device=device, dtype=noise_dtype, generator=g)
        elif self.fixed_per_epoch:
            if self._fixed_noises is None:
                self._fixed_noises = self.noise.sample_batch(B, T, device=device).to(noise_dtype)
            noises = self._fixed_noises.clone()
        else:
            noises = self.noise.sample_batch(B, T, device=device).to(noise_dtype)

        snrs = torch.empty(B, device=device, dtype=torch.float32).uniform_(self.snr_min, self.snr_max)
        new_noisy_bt = mix_at_snr(clean_bt, noises, snrs, peak=self.out_peak)  # (B,T) float32

        # detect aliasing (shared storage between noisy and clean)
        same_mem = False
        try:
            same_mem = (noisy.data_ptr() == clean.data_ptr())
        except Exception:
            same_mem = False

        # write back respecting original shape
        if noisy.dim() == 2:  # (B,T)
            if same_mem:
                if isinstance(batch, dict):
                    batch["noisy"] = new_noisy_bt.to(noisy.dtype if noisy.dtype.is_floating_point else torch.float32)
                elif isinstance(batch, list):
                    batch[0] = new_noisy_bt.to(noisy.dtype if noisy.dtype.is_floating_point else torch.float32)
                else:  # tuple fallback: last resort
                    noisy.copy_(new_noisy_bt.to(noisy.dtype if noisy.dtype.is_floating_point else torch.float32))
            else:
                noisy.copy_(new_noisy_bt.to(noisy.dtype if noisy.dtype.is_floating_point else torch.float32))
        elif noisy.dim() == 3:  # (B,C,T)
            new_noisy_bct = new_noisy_bt.unsqueeze(1).repeat(1, noisy.shape[1], 1)
            new_noisy_bct = new_noisy_bct.to(noisy.dtype if noisy.dtype.is_floating_point else torch.float32)
            if same_mem:
                if isinstance(batch, dict):
                    batch["noisy"] = new_noisy_bct
                elif isinstance(batch, list):
                    batch[0] = new_noisy_bct
                else:
                    noisy.copy_(new_noisy_bct)
            else:
                noisy.copy_(new_noisy_bct)
        else:
            raise RuntimeError(f"Unexpected noisy shape {tuple(noisy.shape)}")

        if self.track:
            self._repl += B

        # ---- one-time post-SNR telemetry (first replaced batch) ----
        try:
            if self.track and self._repl <= B:
                noisy2 = batch["noisy"] if isinstance(batch, dict) else (
                    batch[0] if isinstance(batch, (list, tuple)) else noisy)
                noisy2_bt = _bt(noisy2).to(torch.float32)
                n = min(noisy2_bt.shape[-1], clean_bt.shape[-1])
                num = (clean_bt[..., :n] ** 2).sum(dim=-1).clamp_min(1e-12)
                den = ((noisy2_bt[..., :n] - clean_bt[..., :n]) ** 2).sum(dim=-1).clamp_min(1e-12)
                post_snr = float((10.0 * torch.log10(num / den)).mean().item())
                print(f"[proc-noise] post-mix SNR ~ {post_snr:+.2f} dB")
        except Exception:
            pass
        if self.require_replace:
            # re-read from container in case we reassigned
            noisy2 = batch["noisy"] if isinstance(batch, dict) else (
                batch[0] if isinstance(batch, (list, tuple)) else noisy)
            s = snr_db(_bt(clean), _bt(noisy2))
            if s > 60.0:
                raise RuntimeError(f"ProcNoise require_replace: unmixed batch, SNR={s:+.2f} dB")
