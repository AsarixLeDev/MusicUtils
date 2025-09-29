from __future__ import annotations

import os
import random
from typing import Optional

import torch

from .trainer import Callback
from ..ema.ema import EMA
from ..mining.hard_miner import HardMiner, MutableWeightedSampler
from ..utils.noise import NoiseFactory, mix_at_snr


class EMACallback(Callback):
    """Legacy EMA (update on batch_end; swap for val). Prefer EmaUpdateCallback + EmaEvalSwap."""
    def __init__(self, model, decay=0.999):
        self.ema = EMA(model, decay=float(decay))
        self._backup = None

    def on_batch_end(self, trainer=None, **_):
        self.ema.update(trainer.model)

    def on_val_start(self, trainer=None, **_):
        m = trainer.model
        self._backup = {k: v.detach().clone() for k, v in m.state_dict().items()}
        self.ema.apply_to(m)

    def on_val_end(self, trainer=None, **_):
        if self._backup:
            trainer.model.load_state_dict(self._backup)
            self._backup = None


class CheckpointCallback(Callback):
    def __init__(self, out_dir: str, every: int = 1):
        self.out = out_dir
        self.every = int(every)
        os.makedirs(self.out, exist_ok=True)

    def on_epoch_end(self, trainer=None, state=None, **_):
        if (state.epoch % self.every) != 0:
            return
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
            comps_str = " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(train_comps.items()))
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
        if per_sample is None:
            return
        ids = outputs.get("ids", None) if isinstance(outputs, dict) else None
        self.miner.update_batch(ids, per_sample)

    def on_epoch_end(self, state=None, **_):
        if not self.sampler or state.epoch < self.start_epoch:
            return
        w = self.miner.make_weights(self.dataset)
        self.sampler.set_weights(w)
        print("[hard-mining] sampler weights updated.")


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
        print(f"[proc-noise] epoch {state.epoch}: replaced {self._repl}/{self._seen} items ({frac*100:.1f}%)")
        try:
            trainer.state.info["proc_noise_frac"] = frac
        except Exception:
            pass

class CurriculumCallback(Callback):
    def __init__(self, loss_fn, task, dataset=None,
                 snr_stages=None, sisdr=None, mask_limit=None,
                 mask_variant=None,
                 mask_reg=None,          # <— NEW: schedule weight for mask_unity_reg
                 sisdr_weight=None):     # <— NEW: schedule sisdr_ratio weight
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
            m1 = float(self.mask_limit.get("end",   2.5))
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


class AudioDebugCallback(Callback):
    """
    Save a few (noisy, yhat, clean, resid_true) triads from *validation*,
    but pick items that are actually useful to listen to:
      - not (almost) silent
      - "noisy" has SNR(noisy, clean) below a threshold (i.e., genuinely noisy)

    It samples random indices from the val dataset up to `scan_tries` to find `num` items
    that pass filters, then runs the model once on that mini-batch and saves WAVs.
    """
    def __init__(self, out_dir: str, val_dataset, task, sr=48000,
                 every=5, num_items=3,
                 min_noisy_snr_db: float = 25.0,   # exclude "noisy" that is cleaner than this
                 max_silence_frac: float = 0.95,   # exclude files >95% near-zero
                 scan_tries: int = 400,            # how many random indices to try to find `num_items`
                 seed: int = 1234):
        import os, random
        self.out = os.path.join(out_dir, "audio_debug"); os.makedirs(self.out, exist_ok=True)
        self.ds = val_dataset; self.task = task
        self.sr = int(sr); self.every = int(every); self.num = int(num_items)
        self.min_snr_db = float(min_noisy_snr_db)
        self.max_sil = float(max_silence_frac)
        self.scan_tries = int(scan_tries)
        self._rng = random.Random(seed)

    # small helpers on numpy/torch without imports churn
    def _np_to_ct(self, x):
        import torch, numpy as np
        if isinstance(x, torch.Tensor):
            t = x
        else:
            # x is numpy, make sure it's contiguous
            t = torch.from_numpy(x.copy() if hasattr(x, "flags") and not x.flags.c_contiguous else x)
        t = t.to(torch.float32)
        if t.dim() == 1:  # (T,) -> (1,T)
            t = t.unsqueeze(0)
        elif t.dim() == 2:  # (C,T)
            pass
        else:
            raise RuntimeError(f"Unexpected item shape {tuple(t.shape)}")
        return t.contiguous()

    def _mono_bt(self, t):
        # torch (C,T) or (T,) -> (T,)
        if t.dim() == 2:  # (C,T)
            return t.mean(dim=0)
        return t

    def _rms_dbfs(self, x, eps=1e-12):
        import torch
        rms = torch.sqrt(torch.mean(x**2) + eps)
        return float(20.0 * torch.log10(rms + eps))

    def _sil_frac(self, x, thr_dbfs=-60.0):
        import torch
        thr = 10.0 ** (thr_dbfs / 20.0)
        return float((torch.abs(x) < thr).float().mean())

    def _snr_noisy_clean_db(self, noisy, clean, eps=1e-12):
        import torch
        # both mono 1D, same length
        n = min(noisy.numel(), clean.numel())
        noisy = noisy[:n]; clean = clean[:n]
        num = torch.sum(clean**2).clamp_min(eps)
        den = torch.sum((noisy - clean)**2).clamp_min(eps)
        return float(10.0 * torch.log10(num / den))

    @torch.no_grad()
    def on_epoch_end(self, trainer=None, state=None, **_):
        if (state.epoch % self.every) != 0: return
        try:
            import torchaudio, torch
        except Exception:
            print("[audio-debug] torchaudio/torch not available; skipping")
            return

        # sample candidate indices at random and filter
        N = len(self.ds)
        picks = []
        tries = 0
        while len(picks) < self.num and tries < self.scan_tries:
            i = self._rng.randrange(0, N)
            item = self.ds[i]
            # item expected: (noisy, clean, ...)
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                tries += 1; continue
            noisy_ct = self._np_to_ct(item[0])  # (C,T)
            clean_ct = self._np_to_ct(item[1])  # (C,T)
            # make quick mono 1D
            noisy_m = self._mono_bt(noisy_ct)
            clean_m = self._mono_bt(clean_ct)

            # silence/clean filters
            if self._sil_frac(clean_m) > self.max_sil or self._sil_frac(noisy_m) > self.max_sil:
                tries += 1; continue
            snr = self._snr_noisy_clean_db(noisy_m, clean_m)
            if snr > self.min_snr_db:
                # too clean to be useful listening example
                tries += 1; continue

            picks.append((i, noisy_ct, clean_ct, snr))
            tries += 1

        if not picks:
            print("[audio-debug] no suitable items after scan; saving first few indices as fallback")
            picks = []
            for i in range(min(self.num, len(self.ds))):
                item = self.ds[i]
                if not (isinstance(item, (list, tuple)) and len(item) >= 2): continue
                picks.append((i, self._np_to_ct(item[0]), self._np_to_ct(item[1]), None))

        # stack and run the model once
        dev = next(trainer.model.parameters()).device
        was_train = trainer.model.training
        trainer.model.eval()

        noisy_ct = torch.stack([p[1] for p in picks], dim=0).to(dev, non_blocking=True)   # (B,C,T)
        clean_ct = torch.stack([p[2] for p in picks], dim=0).to(dev, non_blocking=True)   # (B,C,T)
        batch = [noisy_ct, clean_ct]

        with torch.autocast(device_type=("cuda" if dev.type == "cuda" else "cpu"),
                            dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32, enabled=True):
            outputs, _ = self.task.step(trainer.model, batch)
        yhat_bt = outputs["yhat"].detach().to(torch.float32).cpu()   # (B,T) or (B,1,T)->(B,T)

        if yhat_bt.dim() == 3 and yhat_bt.size(1) == 1:
            yhat_bt = yhat_bt[:, 0, :]

        # save files
        for k, (idx, noisy_k, clean_k, snr_k) in enumerate(picks):
            base = os.path.join(self.out, f"ep{state.epoch:03d}_idx{idx:06d}")
            # torchaudio expects (C,T)
            torchaudio.save(base + "_noisy.wav", noisy_k.cpu(), self.sr)
            torchaudio.save(base + "_clean.wav", clean_k.cpu(), self.sr)
            # select yhat by k
            yhat_k = yhat_bt[k].unsqueeze(0)  # (1,T)
            torchaudio.save(base + "_yhat.wav", yhat_k, self.sr)
            # also save true residual (noisy-clean)
            resid_true = (self._mono_bt(noisy_k) - self._mono_bt(clean_k)).unsqueeze(0)
            torchaudio.save(base + "_resid.wav", resid_true, self.sr)

            if snr_k is not None:
                print(f"[audio-debug] saved ep{state.epoch:03d} idx={idx} with SNR(noisy,clean)={snr_k:+.2f} dB")

        if was_train: trainer.model.train()
        print(f"[audio-debug] saved {len(picks)} triads at epoch {state.epoch}")


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


class EpochSeedCallback(Callback):
    def __init__(self, datasets):
        self.datasets = [d for d in (datasets if isinstance(datasets, (list, tuple)) else [datasets]) if d is not None]

    def on_epoch_start(self, state=None, **_):
        seed = 1234 + state.epoch
        import numpy as np, random as _random, torch as _torch
        _random.seed(seed); np.random.seed(seed); _torch.manual_seed(seed)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(state.epoch)


class EmaUpdateCallback(Callback):
    """Keeps EMA weights up-to-date during training."""
    def __init__(self, ema):
        self.ema = ema

    @torch.no_grad()
    def on_batch_end(self, trainer=None, **_):
        self.ema.update(trainer.model)


class EmaEvalSwap(Callback):
    """
    Swap live model weights with EMA for validation, then restore.
    The actual load_state_dict copies are done under `torch.inference_mode()`
    to allow in-place writes on inference-tagged storages.
    """
    def __init__(self, ema_obj, enable: bool = True):
        self.ema = ema_obj
        self.enable = bool(enable)
        self._backup = None
        self._was_training = None

    @torch.no_grad()
    def on_val_start(self, trainer=None, **_):
        if not self.enable or self.ema is None:
            return
        self._was_training = trainer.model.training
        # CPU backup to avoid inference flag issues
        self._backup = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        dev = next(trainer.model.parameters()).device
        ema_sd = {k: v.detach().to(dev) for k, v in self.ema.state_dict().items()}
        with torch.inference_mode():
            trainer.model.load_state_dict(ema_sd, strict=False)
        trainer.model.eval()

    @torch.no_grad()
    def on_val_end(self, trainer=None, **_):
        if self._backup is None:
            return
        dev = next(trainer.model.parameters()).device
        orig_sd = {k: v.to(dev) for k, v in self._backup.items()}
        with torch.inference_mode():
            trainer.model.load_state_dict(orig_sd, strict=False)
        trainer.model.train(self._was_training is True)
        self._backup = None
        self._was_training = None



class DataAuditCallback(Callback):
    """
    Saves a few (noisy, clean, residual) audio triads from TRAIN batches and writes a CSV
    with metrics: duration, RMS dB, SNR dB, silence flags.

    Only touches the first few batches (to keep overhead tiny).
    Place this callback *after* ProcNoiseAugmentCallback in the list so we audit post-augment audio.
    """
    def __init__(self, out_dir: str, sr: int = 48000,
                 first_epochs: int = 1, max_batches: int = 2, max_items: int = 8,
                 silence_rms_db: float = -60.0, save_audio: bool = True, save_csv: bool = True):
        import os
        self.sr = int(sr)
        self.first_epochs = int(first_epochs)
        self.max_batches = int(max_batches)
        self.max_items = int(max_items)
        self.silence_rms_db = float(silence_rms_db)
        self.save_audio = bool(save_audio)
        self.save_csv = bool(save_csv)
        self.root = os.path.join(out_dir, "data_audit")
        os.makedirs(self.root, exist_ok=True)

        # state
        self._epoch_dir = None
        self._batch_count = 0
        self._saved = 0
        self._rows = []

    def _bt(self, t):
        # to (B,T) mono float32 CPU
        if t.dim() == 3:
            t = t.mean(dim=1)
        elif t.dim() == 1:
            t = t.unsqueeze(0)
        return t.detach().to("cpu", dtype=torch.float32, non_blocking=True)

    def _rms_db(self, x, eps=1e-12):
        rms = (x**2).mean(dim=-1).clamp_min(eps).sqrt()
        return 20.0 * torch.log10(rms)

    def _snr_db(self, noisy, clean, eps=1e-12):
        resid = noisy - clean
        num = (clean**2).sum(dim=-1).clamp_min(eps)
        den = (resid**2).sum(dim=-1).clamp_min(eps)
        return 10.0 * torch.log10(num / den)

    @torch.no_grad()
    def on_epoch_start(self, trainer=None, state=None, **_):
        import os
        if state.epoch <= self.first_epochs:
            self._epoch_dir = os.path.join(self.root, f"ep{state.epoch:03d}")
            os.makedirs(self._epoch_dir, exist_ok=True)
            self._batch_count = 0
            self._saved = 0
            self._rows = []
            print(f"[data-audit] will save up to {self.max_items} items from {self.max_batches} batches -> {self._epoch_dir}")
        else:
            self._epoch_dir = None

    @torch.no_grad()
    def on_batch_start(self, trainer=None, state=None, batch=None, **_):
        if self._epoch_dir is None:
            return
        if self._batch_count >= self.max_batches or self._saved >= self.max_items:
            return
        self._batch_count += 1

        try:
            import torchaudio
        except Exception:
            torchaudio = None
            if self.save_audio:
                print("[data-audit] torchaudio not available; will skip audio saving.")

        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            return

        noisy = self._bt(batch[0])
        clean = self._bt(batch[1])
        B, T = noisy.shape
        dur = float(T) / float(self.sr)

        rms_n = self._rms_db(noisy)
        rms_c = self._rms_db(clean)
        snr = self._snr_db(noisy, clean)

        silent_c = (rms_c < self.silence_rms_db)
        silent_n = (rms_n < self.silence_rms_db)

        # rows for CSV
        for i in range(B):
            if self._saved >= self.max_items:
                break
            row = dict(
                epoch=state.epoch, batch=self._batch_count, idx=i,
                dur_s=dur,
                rms_clean_db=float(rms_c[i]),
                rms_noisy_db=float(rms_n[i]),
                snr_db=float(snr[i]),
                clean_silent=bool(silent_c[i]),
                noisy_silent=bool(silent_n[i]),
            )
            self._rows.append(row)

            if self.save_audio and torchaudio is not None:
                base = os.path.join(self._epoch_dir, f"b{self._batch_count:02d}_i{i:02d}")
                resid = (noisy[i] - clean[i]).unsqueeze(0)  # (1,T)
                torchaudio.save(base + "_noisy.wav", noisy[i].unsqueeze(0), self.sr)
                torchaudio.save(base + "_clean.wav", clean[i].unsqueeze(0), self.sr)
                torchaudio.save(base + "_resid.wav", resid, self.sr)
            self._saved += 1

    def on_epoch_end(self, trainer=None, state=None, **_):
        if self._epoch_dir is None:
            return
        import csv, os
        # print quick summary
        import numpy as np
        if self._rows:
            snrs = np.array([r["snr_db"] for r in self._rows], dtype=float)
            rmsc = np.array([r["rms_clean_db"] for r in self._rows], dtype=float)
            sil_rate = float(np.mean([r["clean_silent"] for r in self._rows])) if self._rows else 0.0
            print(f"[data-audit] {len(self._rows)} items | SNR mean {snrs.mean():+.2f} dB (min {snrs.min():+.2f}, max {snrs.max():+.2f}) "
                  f"| clean_silent_rate {sil_rate*100:.1f}% | dur≈{self._rows[0]['dur_s']:.2f}s")
        # write CSV
        if self.save_csv and self._rows:
            csv_path = os.path.join(self._epoch_dir, "audit.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
                w.writeheader()
                for r in self._rows:
                    w.writerow(r)
            print(f"[data-audit] wrote {csv_path}")


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
            if t.dim() == 3: t = t.mean(dim=1)
            elif t.dim() == 1: t = t.unsqueeze(0)
            return t
        noisy_bt = _bt(noisy).to(torch.float32)
        clean_bt = _bt(clean).to(torch.float32)

        B, T = noisy_bt.shape
        self._seen += B

        # SNR(noisy vs clean) ~ 10*log10(||clean||^2 / ||noisy-clean||^2)
        resid = noisy_bt - clean_bt
        num = (clean_bt ** 2).sum(dim=-1).clamp_min(1e-12)
        den = (resid    ** 2).sum(dim=-1).clamp_min(1e-12)
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
                  f"({frac*100:.1f}%) with SNR>{self.min_snr_db} dB")
            try:
                trainer.state.info["min_snr_fixed_frac"] = frac
            except Exception:
                pass

