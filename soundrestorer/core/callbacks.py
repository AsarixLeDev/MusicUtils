from __future__ import annotations

import os
import torch

from .trainer import Callback
from ..utils.noise import NoiseFactory, mix_at_snr

class CurriculumCallback(Callback):
    def __init__(self, loss_fn, task, dataset=None,
                 snr_stages=None, sisdr=None, mask_limit=None,
                 mask_variant=None,
                 mask_reg=None,  # <— NEW: schedule weight for mask_unity_reg
                 sisdr_weight=None):  # <— NEW: schedule sisdr_ratio weight
        self.loss_fn = loss_fn
        self.task = task
        self.dataset = dataset
        self.snr_stages = snr_stages or []
        self.sisdr = sisdr or {}
        self.mask_limit = mask_limit or {}
        self.mask_variant = mask_variant or []  # list of {until: int, variant: str}
        self.mask_reg = mask_reg or {}
        self.sisdr_weight = sisdr_weight or {}

    def _snr_for_epoch(self, epoch):
        # pick first stage whose 'until' >= epoch
        take = None
        for st in self.snr_stages:
            if epoch <= int(st.get("until", 10 ** 9)):
                take = st
                break
        if take is None and self.snr_stages:
            take = self.snr_stages[-1]
        return take

    def on_epoch_start(self, state=None, **_):
        e = state.epoch

        # SNR window
        st = self._snr_for_epoch(e)
        if st:
            if self.dataset is not None:
                for k in ("snr_min", "snr_max", "use_ext_noise_p"):
                    if hasattr(self.dataset, k) and (k in st):
                        setattr(self.dataset, k, st[k])
            print(f"[curriculum] epoch {e}: SNR [{st.get('snr_min', '?')},{st.get('snr_max', '?')}]")

        # SISDR target schedule
        if self.sisdr:
            s0 = float(self.sisdr.get("start_db", 0.0))
            s1 = float(self.sisdr.get("end_db", 12.0))
            e1 = int(self.sisdr.get("end_epoch", 20))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            target = (1.0 - t) * s0 + t * s1
            ok = getattr(self.loss_fn, "set_attr", lambda *a, **k: False)("sisdr_ratio", "min_db", target)
            if ok:
                print(f"[curriculum] epoch {e}: SISDR min_db -> {target:.2f} dB")

        # (optional) bump sisdr_ratio weight slightly after warm-up
        if hasattr(self.loss_fn, "items") and e >= 3:
            for i, (name, w, fn) in enumerate(self.loss_fn.items):
                if name == "sisdr_ratio" and w < 0.40:
                    self.loss_fn.items[i] = (name, 0.40, fn)

        # mask_limit schedule on task
        if self.mask_limit and hasattr(self.task, "mask_limit"):
            m0 = float(self.mask_limit.get("start", 1.5))
            m1 = float(self.mask_limit.get("end", 2.5))
            e1 = int(self.mask_limit.get("end_epoch", 20))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            target = (1.0 - t) * m0 + t * m1
            self.task.mask_limit = target
            print(f"[curriculum] epoch {e}: mask_limit -> {target:.2f}")

        if self.mask_reg:
            w0 = float(self.mask_reg.get("start_w", 0.05))
            w1 = float(self.mask_reg.get("end_w", 0.00))
            e1 = int(self.mask_reg.get("end_epoch", 4))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            w = (1.0 - t) * w0 + t * w1
            if hasattr(self.loss_fn, "items"):
                updated = False
                for i, (name, w_old, fn) in enumerate(self.loss_fn.items):
                    if name == "mask_unity_reg":
                        self.loss_fn.items[i] = (name, w, fn)
                        updated = True
                        break
                if updated:
                    print(f"[curriculum] epoch {e}: mask_unity_reg weight -> {w:.3f}")

            # ramp 'sisdr_ratio' weight up later
        if self.sisdr_weight:
            w0 = float(self.sisdr_weight.get("start_w", 0.10))
            w1 = float(self.sisdr_weight.get("end_w", 0.30))
            e1 = int(self.sisdr_weight.get("end_epoch", 8))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            w = (1.0 - t) * w0 + t * w1
            if hasattr(self.loss_fn, "items"):
                for i, (name, w_old, fn) in enumerate(self.loss_fn.items):
                    if name == "sisdr_ratio":
                        self.loss_fn.items[i] = (name, w, fn)
                        print(f"[curriculum] epoch {e}: sisdr_ratio weight -> {w:.3f}")
                        break

        # mask_variant stage (delta1 -> plain)
        if self.mask_variant and hasattr(self.task, "mask_variant"):
            for stg in self.mask_variant:
                until = int(stg.get("until", 10 ** 9))
                if e <= until:
                    v = str(stg.get("variant", "")).lower()
                    if v:
                        self.task.mask_variant = v
                        print(f"[curriculum] epoch {e}: mask_variant -> {v}")
                    break


class BestCheckpointCallback(Callback):
    def __init__(self, out_dir: str, k: int = 3, monitor: str = "val_loss", mode: str = "min"):
        self.out = out_dir
        os.makedirs(self.out, exist_ok=True)
        self.k = int(k)
        self.monitor = monitor
        self.mode = mode.lower()
        self._best = []  # list of (score, path)

    def _better(self, a, b):
        return (a < b) if self.mode == "min" else (a > b)

    def on_val_end(self, trainer=None, state=None, train_loss=None, val_loss=None, **_):
        score = val_loss if self.monitor == "val_loss" else train_loss
        p = os.path.join(self.out, f"best_ep{state.epoch:03d}.pt")
        torch.save({
            "model": trainer.model.state_dict(),
            "opt": trainer.opt.state_dict(),
            "sched": getattr(trainer.sched, "state_dict", lambda: {})(),
            "epoch": state.epoch, "global_step": state.global_step,
            "score": float(score),
        }, p)
        self._best.append((float(score), p))
        self._best.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self._best) > self.k:
            _, worst_p = self._best.pop()
            try:
                os.remove(worst_p)
            except Exception:
                pass
        if self._best:
            best_p = self._best[0][1]
            link = os.path.join(self.out, "best.pt")
            try:
                if os.path.exists(link):
                    os.remove(link)
            except Exception:
                pass
            try:
                import shutil
                shutil.copyfile(best_p, link)
            except Exception:
                pass


class EarlyStoppingCallback(Callback):
    def __init__(self, patience=10, min_delta=0.0, monitor="val_loss", mode="min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode.lower()
        self.best = None
        self.count = 0
        self.should_stop = False

    def _improved(self, current):
        if self.best is None:
            return True
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def on_val_end(self, train_loss=None, val_loss=None, **_):
        current = val_loss if self.monitor == "val_loss" else train_loss
        if self._improved(current):
            self.best = current
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
                print(f"[early-stop] no improvement for {self.patience} evals. Stopping.")


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
