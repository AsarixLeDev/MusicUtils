# -*- coding: utf-8 -*-
# soundrestorer/core/trainer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from soundrestorer.utils.torch_utils import (
    move_to, autocast_from_amp, need_grad_scaler, set_channels_last,
    load_state_dict_loose, latest_checkpoint, format_ema,
)


# -----------------------
# Simple EMA
# -----------------------

class ModelEMA:
    """
    Exponential Moving Average for a torch.nn.Module.
    Keeps a shadow copy on the same device as the model.
    """

    def __init__(self, model: nn.Module, beta: float = 0.0):
        self.beta = float(beta)
        self.active = self.beta > 0.0
        self.shadow: Optional[nn.Module] = None
        if self.active:
            self.shadow = self._clone(model)

    def _clone(self, model: nn.Module) -> nn.Module:
        copy = type(model)() if hasattr(type(model), "__call__") else nn.Module()
        # generic state copy
        copy.load_state_dict(model.state_dict(), strict=True)
        for p in copy.parameters():
            p.requires_grad_(False)
        copy.to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        return copy

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.active or self.shadow is None:
            return
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        b = self.beta
        for k in ssd.keys():
            ssd[k].lerp_(msd[k], 1.0 - b)

    def state_dict(self):
        if not self.active or self.shadow is None:
            return {}
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        if self.shadow is not None and sd:
            self.shadow.load_state_dict(sd, strict=False)


# -----------------------
# Trainer
# -----------------------

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


class Trainer:
    """
    Orchestrates training for a single Task module (which wraps the model).
    Task forward signature: outputs = task(batch_dict)  -> must include "yhat".
    Loss signature: total, comps = loss_fn(outputs, batch_dict).
    """

    def __init__(
            self,
            *,
            task: nn.Module,
            loss_fn,
            optimizer: Optimizer,
            scheduler: Optional[_LRScheduler],
            run_dir: Path,
            device: torch.device,
            amp: str = "off",
            grad_accum: int = 1,
            grad_clip: float = 0.0,
            channels_last: bool = True,
            compile: bool = False,
            ema_beta: float = 0.0,
            callbacks: Optional[List[Any]] = None,
            print_every: int = 50,
    ):
        self.task = task
        # NEW: legacy alias — some callbacks expect trainer.model
        self.model = self.task

        self.loss_fn = loss_fn
        self.optim = optimizer
        self.sched = scheduler
        self.run_dir = run_dir
        self.device = device
        self.amp = (amp or "off").lower()
        self.grad_accum = max(1, int(grad_accum))
        self.grad_clip = float(grad_clip)
        self.channels_last = bool(channels_last)
        self.compile = bool(compile)
        self.ema = ModelEMA(self.task, ema_beta)
        self.callbacks = callbacks or []
        self.print_every = int(print_every)

        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # placement
        self.task.to(self.device)
        set_channels_last(self.task, self.channels_last)

        # compile if asked (compile the whole task)
        if self.compile and hasattr(torch, "compile"):
            self.task = torch.compile(self.task)  # inductor by default
            print("[compile] torch.compile active (inductor, default)")
        else:
            print("[compile] torch.compile inactive")

        # AMP
        self.autocast = autocast_from_amp(self.amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=need_grad_scaler(self.amp))

        # Train state
        self.state = TrainState(epoch=0, global_step=0)

    # -----------------------
    # Public API
    # -----------------------

    def fit(
            self,
            train_loader: Iterable[Dict[str, torch.Tensor]],
            val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
            *,
            epochs: int,
            save_every: int = 1,
            overfit_steps: int = 0,
    ):
        # callbacks: fit begin
        for cb in self.callbacks:
            fn = getattr(cb, "on_fit_begin", None)
            if callable(fn): fn(self)

        total_epochs = int(epochs)
        steps_per_epoch = (overfit_steps or len(train_loader))
        total_steps = total_epochs * steps_per_epoch

        # One-line summary (mirrors your style)
        print("\n===== TRAINING SUMMARY =====")
        print(f"device={self.device} | amp={self.amp} | channels_last={self.channels_last}")
        n_params = sum(p.numel() for p in self.task.parameters())
        n_trainable = sum(p.numel() for p in self.task.parameters() if p.requires_grad)
        print(f"params: {n_trainable:,} trainable / {n_params:,} total")
        print(f"batch={getattr(train_loader, 'batch_size', '?')} | "
              f"workers={getattr(train_loader, 'num_workers', '?')}")
        print(f"epochs={total_epochs} | steps/epoch={steps_per_epoch} | total_steps={total_steps}")
        print(f"grad_accum={self.grad_accum} | grad_clip={self.grad_clip} | optimizer={type(self.optim).__name__}")
        if self.sched is not None:
            print("scheduler=LR scheduler enabled")
        print(f"ema={format_ema(getattr(self.ema, 'beta', 0.0))}")
        print(f"compile={'on' if self.compile else 'off'} (torch.compile)\n================================\n")

        # main loop
        for epoch in range(self.state.epoch + 1, self.state.epoch + 1 + total_epochs):
            # callbacks: epoch begin
            # Call both new-style and legacy epoch-begin hooks
            for cb in self.callbacks:
                fn = getattr(cb, "on_epoch_begin", None)
                if callable(fn): fn(self, epoch)
                fn_legacy = getattr(cb, "on_epoch_start", None)
                if callable(fn_legacy): fn_legacy(trainer=self, state=self.state)

            start_t = time.time()
            tr_loss, tr_items = self._run_train_epoch(train_loader, epoch, overfit_steps)
            if val_loader is not None:
                for cb in self.callbacks:
                    fn = getattr(cb, "on_val_start", None)
                    if callable(fn): fn(trainer=self, state=self.state)
                val_loss, val_items = self._run_val_epoch(val_loader, epoch)
                for cb in self.callbacks:
                    fn = getattr(cb, "on_val_end", None)
                    if callable(fn):
                        fn(trainer=self, state=self.state,
                           train_loss=tr_loss, val_loss=val_loss,
                           train_comps=None, val_comps=None,
                           train_used=tr_items, train_skipped=0,
                           epoch_time=0.0)
            else:
                val_loss, val_items = (float("nan"), 0)

            elapsed = time.time() - start_t
            lr = self.optim.param_groups[0]["lr"]

            # Epoch report (concise)
            print(f"[epoch {epoch:03d}] train {tr_loss:.4f} | val {val_loss:.4f} | "
                  f"used {tr_items} | lr {lr:.2e} | {elapsed:.1f}s")

            # callbacks: epoch end
            for cb in self.callbacks:
                fn = getattr(cb, "on_epoch_end", None)
                if callable(fn): fn(self, epoch)

            # checkpoint
            if epoch % max(1, int(save_every)) == 0:
                self.save_checkpoint(epoch)

        # callbacks: fit end
        for cb in self.callbacks:
            fn = getattr(cb, "on_fit_end", None)
            if callable(fn): fn(self)

    # -----------------------
    # Internals
    # -----------------------

    def _maybe_zero_grad(self, step_idx: int):
        if step_idx % self.grad_accum == 0:
            self.optim.zero_grad(set_to_none=True)

    def _backward_step(self, loss: torch.Tensor):
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self):
        if self.scaler.is_enabled():
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()

    def _run_train_epoch(
            self,
            loader: Iterable[Dict[str, torch.Tensor]],
            epoch: int,
            overfit_steps: int = 0,
    ) -> Tuple[float, int]:
        self.task.train()
        running = 0.0
        n_items = 0
        step_in_epoch = 0

        if overfit_steps > 0:
            # cache first batch and reuse
            _first = next(iter(loader))
            _first = move_to(_first, self.device)
            cached = _first

        for batch_idx, batch in enumerate(loader):
            if overfit_steps > 0 and batch_idx >= 1:
                batch = cached
            else:
                batch = move_to(batch, self.device)

            # Legacy per-batch start (needed by ProcNoiseAugmentCallback)
            for cb in self.callbacks:
                fn = getattr(cb, "on_batch_start", None)
                if callable(fn):
                    try:
                        fn(trainer=self, state=self.state, batch=batch)
                    except Exception as e:
                        print(f"[cb-warn] on_batch_start error: {e}")
            # forward + loss
            with self.autocast:
                outputs = self.task(batch)
                total_loss, comps = self.loss_fn(outputs, batch)
                # ensure float32 scalar
                if torch.is_tensor(total_loss):
                    loss = total_loss.mean()
                else:
                    loss = torch.as_tensor(total_loss, dtype=torch.float32, device=self.device)

            # backward / step (with grad-accum)
            self._maybe_zero_grad(self.state.global_step + 1)
            self._backward_step(loss / self.grad_accum)

            if self.grad_clip > 0.0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optim)
                clip_grad_norm_(self.task.parameters(), self.grad_clip)

            if (self.state.global_step + 1) % self.grad_accum == 0:
                self._optimizer_step()
                if self.sched is not None:
                    self.sched.step()
                # EMA after optimizer step
                self.ema.update(self.task)

            self.state.global_step += 1
            step_in_epoch += 1

            # running stats
            running += float(loss.detach().item())
            n_items += 1

            # callbacks: train-batch-end
            # new-style batch-end (already present)
            for cb in self.callbacks:
                fn = getattr(cb, "on_train_batch_end", None)
                if callable(fn):
                    try:
                        fn(self, batch_idx, batch, outputs)
                    except Exception as e:
                        print(f"[cb-warn] on_train_batch_end error: {e}")

            # legacy batch-end
            for cb in self.callbacks:
                fn_legacy = getattr(cb, "on_batch_end", None)
                if callable(fn_legacy):
                    try:
                        fn_legacy(trainer=self, state=self.state, batch=batch, outputs=outputs)
                    except Exception as e:
                        print(f"[cb-warn] on_batch_end error: {e}")

            if overfit_steps > 0 and step_in_epoch >= overfit_steps:
                break

            if self.print_every and (batch_idx + 1) % self.print_every == 0:
                lr = self.optim.param_groups[0]["lr"]
                print(f"train {epoch:03d}: {batch_idx + 1}/{overfit_steps or len(loader)} | "
                      f"loss={running / max(1, n_items):.4f} | lr={lr:.2e}")

        return running / max(1, n_items), n_items

    @torch.no_grad()
    def _run_val_epoch(
            self,
            loader: Iterable[Dict[str, torch.Tensor]],
            epoch: int,
    ) -> Tuple[float, int]:
        self.task.eval()
        running = 0.0
        n_items = 0

        for batch_idx, batch in enumerate(loader):
            batch = move_to(batch, self.device)
            with self.autocast:
                outputs = self.task(batch)
                total_loss, _ = self.loss_fn(outputs, batch)
                if torch.is_tensor(total_loss):
                    loss = total_loss.mean()
                else:
                    loss = torch.as_tensor(total_loss, dtype=torch.float32, device=self.device)
            running += float(loss.item())
            n_items += 1

        return running / max(1, n_items), n_items

    # -----------------------
    # Checkpointing
    # -----------------------

    def save_checkpoint(self, epoch: int):
        ckpt = dict(
            epoch=epoch,
            global_step=self.state.global_step,
            task_state=self.task.state_dict(),
            optim_state=self.optim.state_dict(),
            sched_state=(self.sched.state_dict() if self.sched is not None else None),
            ema_state=self.ema.state_dict(),
        )
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, path)
        print(f"[ckpt] saved {path}")

    def resume(self, path: Path) -> Tuple[int, int]:
        ck = latest_checkpoint(path)
        if ck is None:
            raise FileNotFoundError(f"No checkpoint found at {path}")
        sd = torch.load(ck, map_location="cpu")
        # Try loading task (preferred) — contains model submodules.
        tstate = sd.get("task_state", None)
        if tstate:
            load_state_dict_loose(self.task, tstate, "task")
        else:
            # legacy: try model key
            mstate = sd.get("model_state", None)
            if mstate:
                load_state_dict_loose(self.task, mstate, "model")
        # Optimizer / scheduler
        try:
            self.optim.load_state_dict(sd.get("optim_state", {}))
        except Exception as e:
            print(f"[resume] optimizer load warning: {e}")
        if self.sched is not None and sd.get("sched_state", None) is not None:
            try:
                self.sched.load_state_dict(sd["sched_state"])
            except Exception as e:
                print(f"[resume] scheduler load warning: {e}")
        # EMA
        try:
            self.ema.load_state_dict(sd.get("ema_state", {}))
        except Exception as e:
            print(f"[resume] ema load warning: {e}")

        last_epoch = int(sd.get("epoch", 0))
        gstep = int(sd.get("global_step", 0))
        print(f"[resume] loaded {ck} | last_epoch={last_epoch} | global_step={gstep}")
        self.state.epoch = last_epoch
        self.state.global_step = gstep
        return last_epoch, gstep
