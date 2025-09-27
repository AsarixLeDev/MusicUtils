from __future__ import annotations
import math, torch, time
from typing import Dict, Any, List, Optional, Tuple, Iterable
from torch import nn
from .registry import CALLBACKS
from tqdm import tqdm
import time

from .utils import RobustAverager


class TrainState:
    def __init__(self):
        self.epoch = 1
        self.global_step = 0
        self.best = None
        self.info: Dict[str, Any] = {}

class Callback:
    def on_train_start(self, **kw): pass
    def on_epoch_start(self, **kw): pass
    def on_batch_start(self, **kw): pass  # <--- NEW
    def on_batch_end(self, **kw): pass
    def on_val_start(self, **kw): pass
    def on_val_end(self, **kw): pass
    def on_epoch_end(self, **kw): pass
    def on_train_end(self, **kw): pass
    # optional flag for early stopping:
    should_stop: bool = False

class WarmupCosine:
    def __init__(self, optimizer, total_steps: int, warmup: int = 0, min_factor: float = 0.1):
        self.opt = optimizer
        self.tot = max(1,total_steps)
        self.warm = max(0,warmup)
        self.minf = float(min_factor)
        self._step = 0
        self._base = [g['lr'] for g in self.opt.param_groups]

    def step(self):
        self._step += 1
        if self._step <= self.warm:
            s = self._step / max(1, self.warm)
        else:
            prog = (self._step - self.warm) / max(1, self.tot - self.warm)
            s = self.minf + (1 - self.minf) * 0.5 * (1 + math.cos(math.pi * prog))
        for pg, b in zip(self.opt.param_groups, self._base):
            pg['lr'] = b * s

class Trainer:
    """
    Generic trainer that delegates the *step* to a Task object.
    The Task is responsible for turning a batch into model inputs/outputs.
    Losses are computed by ComposedLoss (already assembled).
    """
    def __init__(
        self,
        model: nn.Module,
        task,                 # implements: step(model, batch) -> (outputs, per_sample_proxy)
        loss_fn,              # ComposedLoss-like: returns (scalar, dict of components)
        optimizer,
        runtime,
        scheduler=None,
        device="cuda",
        amp="float32",       # "bfloat16" | "float16" | "float32"
        grad_accum: int = 1,
        grad_clip: float = 0.0,
        channels_last: bool = True,
        compile_model: bool = False,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        if channels_last and device.startswith("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
        self.task = task
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.sched = scheduler
        self.device = device
        self.runtime = runtime
        self.amp = str(amp).lower()
        self.grad_accum = max(1,int(grad_accum))
        self.grad_clip = float(grad_clip)
        self.channels_last = bool(channels_last)
        self.callbacks = callbacks or []
        self.state = TrainState()

        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[compile] torch.compile active")
            except Exception as e:
                print(f"[compile] disabled: {e}")

        # AMP
        self.autocast_dtype = torch.bfloat16 if self.amp == "bfloat16" else (
            torch.float16 if self.amp == "float16" else torch.float32
        )
        self.use_autocast = self.amp in ("bfloat16", "float16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp=="float16"))

        if self.channels_last and (device.startswith("cuda")):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model.to(self.device)

    def _zero(self):
        self.opt.zero_grad(set_to_none=True)

    def _step_sched(self):
        if self.sched is None: return
        # support LambdaLR or our WarmupCosine
        if hasattr(self.sched, "step"): self.sched.step()

    def fit(self, train_loader, val_loader, epochs: int, ckpt_saver=None):
        st = self.state
        robust = RobustAverager(trim_frac=self.loss_cfg.get("train_trim_frac", 0.05))
        clip_thr = float(self.loss_cfg.get("train_loss_clip", 2.0))  # tighten a lot
        for cb in self.callbacks: cb.on_train_start(trainer=self, state=st)

        for epoch in range(st.epoch, epochs + 1):
            st.epoch = epoch
            for cb in self.callbacks: cb.on_epoch_start(trainer=self, state=st)

            self.model.train()
            running = 0.0
            used = 0
            skipped = 0
            step_in_epoch = 0
            comp_sums_tr = {}  # name -> sum over batches
            comp_count_tr = 0

            t0 = time.time()

            use_prefetch = self.device.startswith("cuda") and self.runtime.get("cuda_prefetch", True)
            if use_prefetch:
                from soundrestorer.core.prefetch import CUDAPrefetcher
                prefetch = CUDAPrefetcher(train_loader, device=self.device, channels_last=self.channels_last)
                get_batch = prefetch.next_batch
            else:
                it = iter(train_loader)
                get_batch = lambda: next(it, None)

            batch = get_batch()

            # ----- TRAIN PROGRESS BAR -----
            pbar_tr = tqdm(total=len(train_loader), desc=f"train {epoch:03d}", leave=False, dynamic_ncols=True)

            while batch is not None:
                # callbacks can augment batch (e.g., procedural noise)
                for cb in self.callbacks:
                    cb.on_batch_start(trainer=self, state=st, batch=batch)

                step_in_epoch += 1
                with torch.autocast(device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                                    dtype=self.autocast_dtype, enabled=self.use_autocast):
                    outputs, per_sample = self.task.step(self.model, batch)
                    loss, comps = self.loss_fn(outputs, batch)

                val = float(loss.detach().item())
                if not math.isfinite(val) or (clip_thr > 0 and val > clip_thr):
                    skipped += 1
                    batch = get_batch()
                    continue

                robust.add(val)
                tot += val
                used += 1

                if self.scaler.is_enabled():
                    self.scaler.scale(loss / self.grad_accum).backward()
                    if (st.global_step + 1) % self.grad_accum == 0:
                        self.scaler.unscale_(self.opt)
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self._zero()
                else:
                    (loss / self.grad_accum).backward()
                    if (st.global_step + 1) % self.grad_accum == 0:
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.opt.step()
                        self._zero()

                self._step_sched()
                st.global_step += 1
                used += 1
                running += s

                # aggregate component means (by batch)
                if isinstance(comps, dict):
                    for k, v in comps.items():
                        comp_sums_tr[k] = comp_sums_tr.get(k, 0.0) + float(v)
                    comp_count_tr += 1

                # per-batch callback
                for cb in self.callbacks:
                    cb.on_batch_end(trainer=self, state=st, loss=s, comps=comps, outputs=outputs,
                                    per_sample=per_sample, batch=batch)

                # live bar status
                lr_now = self.opt.param_groups[0]['lr']
                pbar_tr.set_postfix(loss=f"{s:.4f}", avg=f"{(running / max(1, used)):.4f}", lr=f"{lr_now:.2e}")
                pbar_tr.update(1)
                batch = get_batch()


            pbar_tr.close()
            avg_tr = running / max(1, used)
            comps_mean_tr = {k: (comp_sums_tr[k] / max(1, comp_count_tr)) for k in sorted(comp_sums_tr.keys())}
            epoch_time = time.time() - t0

            # ---------- VALIDATION ----------
            self.model.eval()
            tot_v = 0.0
            used_v = 0
            comp_sums_v = {}
            comp_count_v = 0

            for cb in self.callbacks: cb.on_val_start(trainer=self, state=st)

            pbar_va = tqdm(total=len(val_loader), desc=f"valid {epoch:03d}", leave=False, dynamic_ncols=True)
            with torch.no_grad(), torch.autocast(device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                                                 dtype=self.autocast_dtype, enabled=self.use_autocast):
                for batch in val_loader:
                    outputs, _ = self.task.step(self.model, batch)
                    loss_v, comps_v = self.loss_fn(outputs, batch)
                    sv = float(loss_v.detach().item())
                    if not math.isfinite(sv):
                        pbar_va.update(1)
                        continue
                    tot_v += sv
                    used_v += 1
                    if isinstance(comps_v, dict):
                        for k, v in comps_v.items():
                            comp_sums_v[k] = comp_sums_v.get(k, 0.0) + float(v)
                        comp_count_v += 1
                    pbar_va.set_postfix(loss=f"{sv:.4f}")
                    pbar_va.update(1)
            pbar_va.close()

            avg_v = tot_v / max(1, used_v)
            comps_mean_v = {k: (comp_sums_v[k] / max(1, comp_count_v)) for k in sorted(comp_sums_v.keys())}

            # callbacks can log/save using detailed stats
            for cb in self.callbacks:
                cb.on_val_end(trainer=self, state=st, train_loss=avg_tr, val_loss=avg_v,
                              train_comps=comps_mean_tr, val_comps=comps_mean_v,
                              train_used=used, train_skipped=skipped, epoch_time=epoch_time)

            if ckpt_saver:
                ckpt_saver(self.model, self.opt, self.sched, st)

            # final epoch line (also available via ConsoleLogger)
            lr_now = self.opt.param_groups[0]['lr']
            comps_str = " ".join(f"{k}={v:.4f}" for k, v in comps_mean_tr.items())
            print(f"[epoch {epoch:03d}] train {avg_tr:.4f} | val {avg_v:.4f} | used {used} "
                  f"| skip {skipped} | lr {lr_now:.2e} | {epoch_time:.1f}s | {comps_str}")

            for cb in self.callbacks:
                cb.on_epoch_end(trainer=self, state=st, train_loss=avg_tr, val_loss=avg_v,
                                train_comps=comps_mean_tr, val_comps=comps_mean_v,
                                train_used=used, train_skipped=skipped, epoch_time=epoch_time)

        for cb in self.callbacks: cb.on_train_end(trainer=self, state=st)

