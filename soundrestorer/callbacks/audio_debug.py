# -*- coding: utf-8 -*-
# soundrestorer/callbacks/audio_debug.py
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional, Dict

import torch

from soundrestorer.callbacks.callbacks import Callback
from soundrestorer.callbacks.utils import (
    Triad, save_wav_triads, triad_metrics, infer_sr
)
from soundrestorer.metrics.common import match_len


def _slug_from_meta(batch_meta, idx: int) -> str:
    """
    Best-effort: get a short slug from batch['meta']['path'][idx] if present.
    """
    try:
        p = batch_meta.get("path")
        if isinstance(p, (list, tuple)):
            p = p[idx]
        name = Path(str(p)).stem
    except Exception:
        name = ""
    if not name:
        return ""
    # safe slug
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:60]  # keep short


class AudioDebugCallback(Callback):
    """
    Save a few (clean, noisy, yhat, resid) WAV triads every epoch into:
        {run_dir}/logs/audio_debug/epXXX/{base}_{clean,noisy,yhat,resid}.wav

    Selection policy:
      - Collect from the first `scan_batches` train batches each epoch.
      - Save up to `per_epoch` items (one per batch by default).
      - Chooses a single index `idx` per batch (random, reproducible with seed).

    Assumptions:
      - batch dict has keys: "clean" [B,C,T], "noisy" [B,C,T] (noisy optional).
      - outputs dict has key: "yhat" [B,C,T] (waveform).
    """

    def __init__(
            self,
            *,
            per_epoch: int = 3,
            scan_batches: int = 32,
            seed: int = 1234,
            prefer_clean_resid: bool = True,
            subdir: str = "logs/audio_debug",
            sr: Optional[int] = None,
            print_metrics: bool = True,
    ):
        self.per_epoch = int(per_epoch)
        self.scan_batches = int(scan_batches)
        self.seed = int(seed)
        self.prefer_clean_resid = bool(prefer_clean_resid)
        self.subdir = subdir
        self.force_sr = sr
        self.print_metrics = bool(print_metrics)

        self._saved = 0
        self._epoch = 0
        self._rng = random.Random(self.seed)
        self._outdir: Optional[Path] = None

    # ---- hooks (duck-typed; call these from your Trainer) ----

    def on_fit_begin(self, trainer) -> None:
        # determine sr once if not provided
        if self.force_sr is None:
            self.force_sr = infer_sr(trainer)

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        self._epoch = int(epoch)
        self._saved = 0
        run_dir = Path(getattr(trainer, "run_dir", "runs/default"))
        self._outdir = run_dir / self.subdir / f"ep{epoch:03d}"

    def on_train_batch_end(
            self,
            trainer,
            batch_idx: int,
            batch: Dict[str, torch.Tensor],
            outputs: Dict[str, torch.Tensor],
    ) -> None:
        if self._saved >= self.per_epoch:
            return
        if batch_idx >= self.scan_batches:
            return

        yhat = outputs.get("yhat", None)
        clean = batch.get("clean", None)
        noisy = batch.get("noisy", None)
        if yhat is None or clean is None:
            print("[WARNING] yhat or clean missing")
            return

        B = yhat.shape[0] if yhat.dim() == 3 else 1
        idx = self._rng.randrange(0, B)

        y_i = yhat[idx:idx + 1]  # [1,C,T]
        c_i = clean[idx:idx + 1] if clean is not None else None
        n_i = noisy[idx:idx + 1] if noisy is not None else None

        # collapse batch dim
        if y_i.dim() == 3: y_i = y_i[0]
        if c_i is not None and c_i.dim() == 3: c_i = c_i[0]
        if n_i is not None and n_i.dim() == 3: n_i = n_i[0]

        # ensure same length
        if c_i is not None:
            y_i, c_i = match_len(y_i, c_i)
        if n_i is not None:
            y_i, n_i = match_len(y_i, n_i)

        # sample rate
        sr = self.force_sr or infer_sr(trainer) or 48000

        slug = ""
        meta = batch.get("meta")
        if isinstance(meta, dict):
            slug = _slug_from_meta(meta, idx)

        if slug:
            base = f"{slug}_b{batch_idx}_i{idx}"
        else:
            base = f"b{batch_idx}_i{idx}"
        paths = save_wav_triads(
            self._outdir, base,
            Triad(clean=c_i, noisy=n_i, yhat=y_i, sr=sr),
            want_resid=True
        )

        self._saved += 1

        # quick metrics (optional print)
        if self.print_metrics:
            met = triad_metrics(y_i, c_i, n_i)
            snr = met.get("snr_noisy_clean_db", None)
            if snr is not None:
                print(f"[audio-debug] saved ep{self._epoch:03d} {base} with SNR(noisy,clean)={float(snr):+.2f} dB")
            else:
                print(f"[audio-debug] saved ep{self._epoch:03d} {base}")

    def on_epoch_end(self, trainer, epoch: int) -> None:
        if self._saved > 0:
            print(f"[audio-debug] saved {self._saved} triads at epoch {epoch}")
