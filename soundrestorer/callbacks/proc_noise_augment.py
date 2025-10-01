from soundrestorer.callbacks.callbacks import Callback
from ..utils.noise import NoiseFactory, mix_at_snr

import torch


class ProcNoiseAugmentCallback(Callback):
    """
    Replaces batch noisy = clean + procedural_noise at random, on-the-fly.
    Works without modifying your dataset. Train-only by default.
    Now tracks how many items were replaced to report the true fraction.
    """

    def __init__(self, sr, prob=0.5, snr_min=0.0, snr_max=20.0, out_peak=0.98,
                 train_only=True, noise_cfg=None, track_stats: bool = True):
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

    def on_epoch_start(self, **_):
        self._seen = 0
        self._repl = 0

    def on_batch_start(self, trainer=None, state=None, batch=None, **_):
        if self.train_only and trainer.model.training is False:
            return
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            return
        noisy, clean = batch[0], batch[1]
        device = clean.device
        B, T = clean.shape[0], clean.shape[-1]
        if self.track:
            self._seen += B

        import random
        # coin toss per-batch: if success, rebuild entire noisy batch
        if random.random() > self.prob:
            return

        # procedural noise
        noises = self.noise.sample_batch(B, T, device=device)  # (B,T)
        snrs = torch.empty(B, device=device).uniform_(self.snr_min, self.snr_max)
        new_noisy_bt = mix_at_snr(clean, noises, snrs, peak=self.out_peak)  # (B,T)

        # broadcast to match original 'noisy' shape and do in-place copy
        if noisy.dim() == 2:  # (B,T)
            noisy.copy_(new_noisy_bt)
        elif noisy.dim() == 3:  # (B,C,T)
            noisy.copy_(new_noisy_bt.unsqueeze(1).expand(-1, noisy.shape[1], -1))
        else:
            raise RuntimeError(f"Unexpected noisy shape {tuple(noisy.shape)}")

        if self.track:
            self._repl += B

    def on_epoch_end(self, trainer=None, state=None, **_):
        if not self.track or self._seen == 0:
            return
        frac = self._repl / float(self._seen)
        print(f"[proc-noise] epoch {state.epoch}: replaced {self._repl}/{self._seen} items ({frac * 100:.1f}%)")
        try:
            trainer.state.info["proc_noise_frac"] = frac
        except Exception:
            pass