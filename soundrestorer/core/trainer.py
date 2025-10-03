# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# optional utilities if present in your project (we guard their absence)
try:
    from soundrestorer.utils.metrics import si_sdr_db  # robust SI-SDR (dB)
except Exception:
    si_sdr_db = None  # we'll skip SI prints if missing


# =========================
# State, Callbacks, Scheds
# =========================

class TrainState:
    def __init__(self):
        self.epoch: int = 1
        self.global_step: int = 0
        self.best: Optional[float] = None
        self.info: Dict[str, Any] = {}


class Callback:
    def on_fit_begin(self, **kw): pass
    def on_fit_end(self, **kw): pass

    def on_epoch_begin(self, **kw): pass
    def on_epoch_end(self, **kw): pass

    def on_batch_start(self, **kw): pass
    def on_batch_end(self, **kw): pass

    def on_val_begin(self, **kw): pass
    def on_val_end(self, **kw): pass

    # optional early stop flag
    should_stop: bool = False


class WarmupCosine:
    """Simple warmup+cosine schedule (optional; not required for constant-LR overfit)."""
    def __init__(self, optimizer, total_steps: int, warmup: int = 0, min_factor: float = 0.1):
        self.opt = optimizer
        self.tot = max(1, int(total_steps))
        self.warm = max(0, int(warmup))
        self.minf = float(min_factor)
        self._step = 0
        self._base = [g["lr"] for g in self.opt.param_groups]

    def step(self):
        self._step += 1
        if self._step <= self.warm:
            s = self._step / max(1, self.warm)
        else:
            prog = (self._step - self.warm) / max(1, self.tot - self.warm)
            s = self.minf + (1 - self.minf) * 0.5 * (1 + math.cos(math.pi * prog))
        for pg, b in zip(self.opt.param_groups, self._base):
            pg["lr"] = b * s


# =========
# Trainer
# =========

BatchType = Union[Dict[str, Any], List[Any], Tuple[Any, ...]]

class Trainer:
    # REPLACE your current __init__ signature with this one
    def __init__(
            self,
            model: nn.Module,
            task: nn.Module,  # callable: outputs = task(batch)
            loss_fn: nn.Module,  # returns (loss_scalar, comps_dict)
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[Any] = None,
            device: Optional[torch.device] = None,
            amp: str = "float32",
            grad_accum: int = 1,
            grad_clip: float = 0.0,
            channels_last: bool = True,
            callbacks: Optional[List[Callback]] = None,
            # ---- NEW (optional, tolerated) ----
            run_dir: Optional[str] = None,
            compile: Optional[bool] = None,  # just stored; not used by this class
            ema_beta: Optional[float] = None,  # stored for callbacks that might read it
            print_every: Optional[int] = None,
            **unused,  # swallow any other legacy kwargs safely
    ):
        self.model = model
        self.task = task
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.sched = scheduler
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.amp = str(amp).lower()
        self.grad_accum = max(1, int(grad_accum))
        self.grad_clip = float(grad_clip)
        self.channels_last = bool(channels_last)
        self.cbs: List[Callback] = callbacks or []
        self.state = TrainState()

        # external cfg if scripts set trainer.cfg elsewhere
        self.cfg: Dict[str, Any] = getattr(self, "cfg", {})

        # ---- store new extras (optional) ----
        self.run_dir = run_dir or getattr(self, "run_dir", None)
        self.compile_requested = bool(compile) if compile is not None else None
        self.ema_beta = float(ema_beta) if ema_beta is not None else None
        self._print_every_override = int(print_every) if print_every is not None else None

        # AMP + scaler (fp16 only)
        self._autocast_enabled = self.amp in ("float16",)
        self._autocast_dtype = (
            torch.float16 if self.amp == "float16"
            else (torch.bfloat16 if self.amp in ("bfloat16", "bf16") else torch.float32)
        )
        self._scaler = torch.cuda.amp.GradScaler(enabled=(self.amp == "float16"))

        # memory format + device move
        if self.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.to(self.device)

    # -------------
    # small helpers
    # -------------

    @staticmethod
    def _bt(x: torch.Tensor) -> torch.Tensor:
        """(B,C,T)->(B,T) mono; (B,T)->(B,T); (T,)->(1,T)."""
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        if x.dim() == 1:
            return x.unsqueeze(0)
        raise RuntimeError(f"Unexpected waveform shape {tuple(x.shape)}")

    def _clone_batch(self, batch: BatchType) -> BatchType:
        """Deep-ish clone tensors so mutators can safely modify."""
        if isinstance(batch, dict):
            return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            out = []
            for v in batch:
                out.append(v.detach().clone() if torch.is_tensor(v) else v)
            return out
        return batch

    def _to_device(self, batch: BatchType) -> BatchType:
        """Move tensors to the trainer device."""
        if isinstance(batch, dict):
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device, non_blocking=True)
            return batch
        if isinstance(batch, (list, tuple)):
            out = []
            for v in batch:
                out.append(v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            return out
        return batch

    # -------------
    # public API
    # -------------

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        overfit_steps: Optional[int] = None,
    ):
        dbg = self.cfg.get("debug", {}) if isinstance(self.cfg, dict) else {}
        pbar_every = int(dbg.get("pbar_every", 50))
        print_comp = bool(dbg.get("print_comp", True))
        print_sisdr = bool(dbg.get("print_sisdr", False))
        val_every = int(dbg.get("val_every", 1))
        val_max_batches = int(dbg.get("val_max_batches", 0))
        of_steps = int(dbg.get("overfit_steps", overfit_steps or 0))

        # callbacks: fit begin
        for cb in self.cbs:
            try:
                cb.on_fit_begin(trainer=self, state=self.state)
            except Exception as e:
                print(f"[cb-warn] on_fit_begin: {e}")

        for epoch in range(1, epochs + 1):
            self.state.epoch = epoch
            used, tr_loss, tr_comps, tr_time = self._run_train_epoch(
                train_loader, epoch, of_steps, pbar_every, print_comp, print_sisdr
            )

            # ---- validation ----
            va_loss = None
            if (val_loader is not None) and (val_every <= 1 or (epoch % val_every) == 0):
                print(f"----- VALIDATION {epoch:03d} -----")
                va_loss = self._run_valid_epoch(val_loader, val_max_batches)
                print(f"[val-done {epoch:03d}] avg_loss={va_loss:.4f} | used={1 if val_max_batches==1 else 'auto'}")

            # callbacks: epoch end
            for cb in self.cbs:
                try:
                    cb.on_epoch_end(trainer=self, state=self.state,
                                    train_loss=tr_loss, train_comps=tr_comps,
                                    val_loss=va_loss, epoch_time=tr_time)
                except Exception as e:
                    print(f"[cb-warn] on_epoch_end: {e}")

            # early stop?
            if any(getattr(cb, "should_stop", False) for cb in self.cbs):
                print("[trainer] early stop requested by callback.")
                break

            print(f"[train-done {epoch:03d}] used {used} steps | avg_loss={tr_loss:.4f} | time={tr_time:.1f}s | lr={self.opt.param_groups[0]['lr']:.2e}")

        for cb in self.cbs:
            try:
                cb.on_fit_end(trainer=self, state=self.state)
            except Exception as e:
                print(f"[cb-warn] on_fit_end: {e}")

    # -----------------
    # internal epochs
    # -----------------

    def _run_train_epoch(
        self,
        loader,
        epoch: int,
        overfit_steps: int,
        pbar_every: int,
        print_comp: bool,
        print_sisdr: bool,
    ) -> Tuple[int, float, Dict[str, float], float]:
        self.model.train(True)

        # one-time cached batch for true overfit
        cached = None
        steps_goal: int
        if overfit_steps and overfit_steps > 0:
            try:
                first = next(iter(loader))
            except StopIteration:
                raise RuntimeError("Empty train loader in overfit mode.")
            cached = self._to_device(self._clone_batch(first))
            steps_goal = overfit_steps
            print(f"[overfit] repeating the same batch for {steps_goal} steps.")
        else:
            steps_goal = len(loader)

        t0 = time.time()
        used = 0
        total_loss = 0.0
        comps_accum: Dict[str, float] = {}

        # allow callbacks to prepare the epoch
        for cb in self.cbs:
            try:
                cb.on_epoch_begin(trainer=self, state=self.state)
            except Exception as e:
                print(f"[cb-warn] on_epoch_begin: {e}")

        loader_iter = iter(loader)

        for step in range(steps_goal):
            # prepare batch
            if cached is not None:
                batch = self._clone_batch(cached)
            else:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                batch = self._to_device(batch)

            # on_batch_start (mutators can modify batch here)
            for cb in self.cbs:
                try:
                    cb.on_batch_start(trainer=self, state=self.state, batch=batch)
                except Exception as e:
                    print(f"[cb-warn] on_batch_start error: {e}")

            # forward + loss
            with torch.autocast(
                device_type=("cuda" if self.device.type == "cuda" else "cpu"),
                dtype=self._autocast_dtype,
                enabled=self._autocast_enabled,
            ):
                outputs = self.task(batch)
                loss, comps = self.loss_fn(outputs, batch)

            # backward
            if self._autocast_enabled:
                self._scaler.scale(loss / self.grad_accum).backward()
                if (self.state.global_step + 1) % self.grad_accum == 0:
                    if self.grad_clip > 0:
                        self._scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self._scaler.step(self.opt)
                    self._scaler.update()
                    self.opt.zero_grad(set_to_none=True)
            else:
                (loss / self.grad_accum).backward()
                if (self.state.global_step + 1) % self.grad_accum == 0:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

            # track
            used += 1
            self.state.global_step += 1
            l_val = float(loss.detach().to("cpu"))
            total_loss += l_val
            if isinstance(comps, dict):
                for k, v in comps.items():
                    try:
                        comps_accum[k] = comps_accum.get(k, 0.0) + float(v)
                    except Exception:
                        pass

            # on_batch_end (pass outputs/loss for savers)
            for cb in self.cbs:
                try:
                    cb.on_batch_end(trainer=self, state=self.state, batch=batch,
                                    outputs=outputs, loss=l_val)
                except Exception as e:
                    print(f"[cb-warn] on_batch_end error: {e}")

            # progress line
            if pbar_every and ((step + 1) % pbar_every == 0 or (step + 1) == steps_goal):
                # grad norm (best-effort)
                try:
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
                    grad_norm_str = f"{float(gn):.3e}"
                except Exception:
                    grad_norm_str = "n/a"

                # L1(n,c) vs L1(y,c)
                l1_str = ""
                try:
                    noisy, clean = None, None
                    if isinstance(batch, dict):
                        noisy, clean = batch.get("noisy", None), batch.get("clean", None)
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        noisy, clean = batch[0], batch[1]
                    if noisy is not None and clean is not None and torch.is_tensor(outputs.get("yhat", None)):
                        nb = self._bt(noisy); cb = self._bt(clean); yb = self._bt(outputs["yhat"])
                        nT = min(nb.shape[-1], cb.shape[-1], yb.shape[-1])
                        mae_n = torch.mean(torch.abs(nb[..., :nT] - cb[..., :nT])).item()
                        mae_y = torch.mean(torch.abs(yb[..., :nT] - cb[..., :nT])).item()
                        l1_str = f" | L1(n,c)={mae_n:.4f} -> L1(y,c)={mae_y:.4f}"
                except Exception:
                    pass

                # SI-SDR delta (optional)
                sisdr_str = ""
                if print_sisdr and si_sdr_db is not None:
                    try:
                        nb = self._bt(noisy); cbv = self._bt(clean); yb = self._bt(outputs["yhat"])
                        nT = min(nb.shape[-1], cbv.shape[-1], yb.shape[-1])
                        si_n = si_sdr_db(nb[..., :nT], cbv[..., :nT]).mean().item()
                        si_y = si_sdr_db(yb[..., :nT], cbv[..., :nT]).mean().item()
                        sisdr_str = f" | SI {si_n:+.2f}→{si_y:+.2f} dB"
                    except Exception:
                        pass

                comp_str = ""
                if print_comp and isinstance(comps, dict) and comps:
                    try:
                        comp_str = " | " + " ".join(f"{k}={float(v):.4f}" for k, v in sorted(comps.items()))
                    except Exception:
                        pass
                else:
                    print(type(comps), comp_str)

                lr = self.opt.param_groups[0]["lr"]
                print(f"[train {epoch:03d}] step {step+1:5d}/{steps_goal} | loss={l_val:.4f}{l1_str}{sisdr_str}{comp_str} | lr={lr:.2e} | grad_norm={grad_norm_str}")

            # step scheduler if any
            if self.sched is not None:
                try:
                    self.sched.step()
                except Exception:
                    pass

        dt = time.time() - t0
        avg_loss = total_loss / max(1, used)
        avg_comps = {k: v / max(1, used) for k, v in comps_accum.items()}
        return used, avg_loss, avg_comps, dt

    @torch.no_grad()
    def _run_valid_epoch(self, val_loader, val_max_batches: int) -> float:
        self.model.train(False)
        for cb in self.cbs:
            try:
                cb.on_val_begin(trainer=self, state=self.state)
            except Exception as e:
                print(f"[cb-warn] on_val_begin: {e}")

        total = 0.0
        used = 0
        it = iter(val_loader)
        steps = val_max_batches if val_max_batches and val_max_batches > 0 else len(val_loader)

        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = self._to_device(batch)

            with torch.autocast(
                device_type=("cuda" if self.device.type == "cuda" else "cpu"),
                dtype=self._autocast_dtype,
                enabled=self._autocast_enabled,
            ):
                outputs = self.task(batch)
                loss, comps = self.loss_fn(outputs, batch)

            total += float(loss.detach().to("cpu"))
            used += 1

        va_loss = total / max(1, used)

        for cb in self.cbs:
            try:
                cb.on_val_end(trainer=self, state=self.state, val_loss=va_loss)
            except Exception as e:
                print(f"[cb-warn] on_val_end: {e}")

        return va_loss
