from __future__ import annotations
import os, torch
from typing import Dict, Any, List, Optional
from .trainer import Callback
from ..ema.ema import EMA
from ..mining.hard_miner import HardMiner, MutableWeightedSampler
import math, random, os, torch
from .trainer import Callback
from ..utils.noise import NoiseFactory, mix_at_snr

class EMACallback(Callback):
    def __init__(self, model, decay=0.999):
        self.ema = EMA(model, decay=float(decay))
        self._backup = None
    def on_batch_end(self, **kw):
        self.ema.update(kw["trainer"].model)
    def on_val_start(self, trainer=None, **_):
        # swap weights
        m = trainer.model
        self._backup = {k: v.detach().clone() for k,v in m.state_dict().items()}
        self.ema.apply_to(m)
    def on_val_end(self, trainer=None, **_):
        # restore
        if self._backup:
            trainer.model.load_state_dict(self._backup); self._backup=None

class CheckpointCallback(Callback):
    def __init__(self, out_dir: str, every: int = 1):
        self.out = out_dir; self.every = int(every)
        os.makedirs(self.out, exist_ok=True)
    def on_epoch_end(self, trainer=None, state=None, **_):
        if (state.epoch % self.every) != 0: return
        p = os.path.join(self.out, f"epoch_{state.epoch:03d}.pt")
        torch.save({
            "model": trainer.model.state_dict(),
            "opt": trainer.opt.state_dict(),
            "sched": getattr(trainer.sched, "state_dict", lambda: {})(),
            "epoch": state.epoch, "global_step": state.global_step
        }, p)
        print(f"[ckpt] saved {p}")

class ConsoleLogger(Callback):
    def on_val_end(self, state=None, train_loss=None, val_loss=None,
                   train_comps=None, val_comps=None, train_used=None,
                   train_skipped=None, epoch_time=None, trainer=None, **_):
        lr_now = trainer.opt.param_groups[0]['lr'] if trainer else 0.0
        comps_str = ""
        if isinstance(train_comps, dict) and train_comps:
            comps_str = " | " + " ".join(f"{k}={v:.4f}" for k,v in sorted(train_comps.items()))
        print(f"[epoch {state.epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f} "
              f"| used {train_used} | skip {train_skipped} | lr {lr_now:.2e} "
              f"| {epoch_time:.1f}s{comps_str}")

class HardMiningCallback(Callback):
    """
    Uses per-sample proxy from task (or computes L1) to update a mutable weighted sampler.
    """
    def __init__(self, dataset, sampler: Optional[MutableWeightedSampler],
                 start_epoch=3, ema=0.9, top_frac=0.3, boost=4.0, baseline=1.0):
        self.dataset = dataset
        self.sampler = sampler
        self.start_epoch = int(start_epoch)
        self.miner = HardMiner(ema=ema, top_frac=top_frac, boost=boost, baseline=baseline)

    def on_batch_end(self, per_sample=None, outputs=None, **_):
        if per_sample is None: return
        ids = outputs.get("ids", None)  # task can supply ids
        self.miner.update_batch(ids, per_sample)

    def on_epoch_end(self, state=None, **_):
        if not self.sampler or state.epoch < self.start_epoch: return
        w = self.miner.make_weights(self.dataset)
        self.sampler.set_weights(w)
        print("[hard-mining] sampler weights updated.")


class ProcNoiseAugmentCallback(Callback):
    """
    Replaces batch noisy = clean + procedural_noise at random, on-the-fly.
    Works without modifying your dataset. Train-only by default.
    """
    def __init__(self, sr, prob=0.5, snr_min=0.0, snr_max=20.0, out_peak=0.98,
                 train_only=True, noise_cfg=None):
        self.sr = int(sr)
        self.prob = float(prob)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.out_peak = float(out_peak)
        self.train_only = bool(train_only)
        self.noise = NoiseFactory(self.sr, noise_cfg or {})

    def on_batch_start(self, trainer=None, state=None, batch=None, **_):
        if self.train_only and trainer.model.training is False: return
        if random.random() > self.prob: return
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2): return

        noisy, clean = batch[0], batch[1]
        device = clean.device
        B, T = clean.shape[0], clean.shape[-1]
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


class CurriculumCallback(Callback):
    """
    Schedules:
      - dataset or proc-noise SNR window,
      - SISDR target (loss),
      - task.mask_limit (stability/expressivity).
    """
    def __init__(self, loss_fn, task, dataset=None,
                 snr_stages=None, sisdr=None, mask_limit=None,
                 mask_variant=None):               # <— NEW
        self.loss_fn = loss_fn
        self.task = task
        self.dataset = dataset
        self.snr_stages = snr_stages or []
        self.sisdr = sisdr or {}
        self.mask_limit = mask_limit or {}
        self.mask_variant = mask_variant or []     # list of {until: int, variant: str}

    def _snr_for_epoch(self, epoch):
        # pick last stage whose 'until' >= epoch
        take = None
        for st in self.snr_stages:
            if epoch <= int(st.get("until", 1e9)):
                take = st; break
        if take is None and self.snr_stages:
            take = self.snr_stages[-1]
        return take

    def on_epoch_start(self, state=None, **_):
        e = state.epoch
        # SNR window
        st = self._snr_for_epoch(e)
        if st:
            snr_min = float(st.get("snr_min", 0.0))
            snr_max = float(st.get("snr_max", 20.0))
            # if dataset exposes these, set them (best effort)
            if self.dataset is not None:
                for k in ("snr_min","snr_max","use_ext_noise_p"):
                    if hasattr(self.dataset, k) and (k in st):
                        setattr(self.dataset, k, st[k])
            # also set onto task if it has proc-noise callback bound (optional)
            # (No direct link here; proc-noise callback can be created with wide range.)
            print(f"[curriculum] epoch {e}: SNR [{snr_min},{snr_max}]")

        # SISDR target schedule
        if self.sisdr:
            s0 = float(self.sisdr.get("start_db", 0.0))
            s1 = float(self.sisdr.get("end_db", 12.0))
            e1 = int(self.sisdr.get("end_epoch", 20))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            target = (1.0 - t) * s0 + t * s1
            ok = self.loss_fn.set_attr("sisdr_ratio", "min_db", target)
            if ok:
                print(f"[curriculum] epoch {e}: SISDR min_db -> {target:.2f} dB")

        # mask_limit schedule on task
        if self.mask_limit:
            m0 = float(self.mask_limit.get("start", 1.5))
            m1 = float(self.mask_limit.get("end",   2.5))
            e1 = int(self.mask_limit.get("end_epoch", 20))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            target = (1.0 - t) * m0 + t * m1
            if hasattr(self.task, "mask_limit"):
                self.task.mask_limit = target
                print(f"[curriculum] epoch {e}: mask_limit -> {target:.2f}")

        for st in self.mask_variant:
            if e <= int(st.get("until", 10 ** 9)):
                v = str(st.get("variant", "")).lower()
                if v and hasattr(self.task, "mask_variant"):
                    self.task.mask_variant = v
                    # Re-bias head to identity for the new variant
                    try:
                        from soundrestorer.models.init_utils import init_head_for_mask
                        init_head_for_mask(trainer.model, v)
                    except Exception:
                        pass
                break


class AudioDebugCallback(Callback):
    def __init__(self, out_dir: str, val_dataset, task, sr=48000, every=5, num_items=3):
        self.out = os.path.join(out_dir, "audio_debug"); os.makedirs(self.out, exist_ok=True)
        self.ds = val_dataset; self.task = task
        self.sr = int(sr); self.every = int(every); self.num = int(num_items)

    @torch.no_grad()
    def on_epoch_end(self, trainer=None, state=None, **_):
        if (state.epoch % self.every) != 0: return
        try:
            import torchaudio
        except Exception:
            print("[audio-debug] torchaudio not available; skipping")
            return

        model = trainer.model
        dev = next(model.parameters()).device  # <- robust device
        was_train = model.training
        model.eval()

        # fixed small subset (0..num-1)
        idx = list(range(min(self.num, len(self.ds))))
        items = [self.ds[i] for i in idx]

        def _np_to_ct(x):
            # x is numpy (C,T) or (T,) -> torch (C,T)
            if isinstance(x, torch.Tensor):
                t = x
            else:
                t = torch.from_numpy(x.copy() if hasattr(x, "flags") and not x.flags.c_contiguous else x)
            t = t.to(torch.float32)
            if t.dim() == 1:  # (T,) -> (1,T)
                t = t.unsqueeze(0)
            return t.contiguous()

        noisy_ct = [_np_to_ct(it[0]) for it in items]  # list of (C,T)
        clean_ct = [_np_to_ct(it[1]) for it in items]

        noisy = torch.stack(noisy_ct, dim=0).to(dev, non_blocking=True)  # (B,C,T)
        clean = torch.stack(clean_ct, dim=0).to(dev, non_blocking=True)  # (B,C,T)

        # Build a batch the Task accepts: (noisy, clean, ...)
        batch = [noisy, clean]

        with torch.autocast(device_type=("cuda" if dev.type == "cuda" else "cpu"),
                            dtype=torch.bfloat16, enabled=True):
            outputs, _ = self.task.step(model, batch)
        yhat = outputs["yhat"].detach().cpu()  # (B,T)

        for i in range(yhat.shape[0]):
            base = os.path.join(self.out, f"ep{state.epoch:03d}_idx{i}")
            # torchaudio.save expects (C,T)
            torchaudio.save(base + "_noisy.wav", noisy[i].cpu(), self.sr)  # (C,T)
            torchaudio.save(base + "_yhat.wav", yhat[i].unsqueeze(0), self.sr)  # (1,T)
            torchaudio.save(base + "_clean.wav", clean[i].cpu(), self.sr)  # (C,T)

        if was_train: model.train()
        print(f"[audio-debug] saved {self.num} triads at epoch {state.epoch}")


class BestCheckpointCallback(Callback):
    def __init__(self, out_dir: str, k: int = 3, monitor: str = "val_loss", mode: str = "min"):
        self.out = out_dir; os.makedirs(self.out, exist_ok=True)
        self.k = int(k); self.monitor = monitor; self.mode = mode.lower()
        self._best = []  # list of (score, path)

    def _better(self, a, b):
        return (a < b) if self.mode == "min" else (a > b)

    def on_val_end(self, trainer=None, state=None, train_loss=None, val_loss=None, **_):
        score = val_loss if self.monitor == "val_loss" else train_loss
        # save current
        p = os.path.join(self.out, f"best_ep{state.epoch:03d}.pt")
        torch.save({
            "model": trainer.model.state_dict(),
            "opt": trainer.opt.state_dict(),
            "sched": getattr(trainer.sched, "state_dict", lambda: {})(),
            "epoch": state.epoch, "global_step": state.global_step,
            "score": float(score),
        }, p)
        self._best.append((float(score), p))
        # keep top-k
        self._best.sort(key=lambda x: x[0], reverse=(self.mode=="max"))
        while len(self._best) > self.k:
            _, worst_p = self._best.pop()
            try: os.remove(worst_p)
            except Exception: pass
        # write pointer to overall best
        if self._best:
            best_p = self._best[0][1]
            link = os.path.join(self.out, "best.pt")
            try:
                if os.path.exists(link): os.remove(link)
            except Exception: pass
            try:
                import shutil; shutil.copyfile(best_p, link)
            except Exception: pass

class EarlyStoppingCallback(Callback):
    def __init__(self, patience=10, min_delta=0.0, monitor="val_loss", mode="min"):
        self.patience = int(patience); self.min_delta = float(min_delta)
        self.monitor = monitor; self.mode = mode.lower()
        self.best = None; self.count = 0
        self.should_stop = False

    def _improved(self, current):
        if self.best is None: return True
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def on_val_end(self, train_loss=None, val_loss=None, **_):
        current = val_loss if self.monitor == "val_loss" else train_loss
        if self._improved(current):
            self.best = current; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
                print(f"[early-stop] no improvement for {self.patience} evals. Stopping.")

class EpochSeedCallback(Callback):
    def __init__(self, datasets):
        self.datasets = [d for d in (datasets if isinstance(datasets, (list,tuple)) else [datasets]) if d is not None]
    def on_epoch_start(self, state=None, **_):
        seed = 1234 + state.epoch
        import numpy as np, random, torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"): ds.set_epoch(state.epoch)


