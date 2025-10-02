# soundrestorer/callbacks/save_preds.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# be tolerant to both callback bases
try:
    from soundrestorer.callbacks.callbacks import Callback
except Exception:
    from soundrestorer.core.trainer import Callback  # legacy

from soundrestorer.callbacks.utils import (
    Triad, save_wav_triads, triad_metrics, infer_sr
)
from soundrestorer.metrics.common import match_len, silence_fraction
from soundrestorer.utils.metrics import rms_db


def _as_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return float("nan")


def _slug_from_meta(meta: Any, idx: int) -> str:
    """
    Produce a short filesystem-safe slug from a batch metadata element.
    Prefers meta['path'] / ['clean_path'] / ['noisy_path'] if present.
    """
    cand: Optional[str] = None
    try:
        if isinstance(meta, dict):
            for k in ("path", "clean_path", "noisy_path", "track"):
                v = meta.get(k)
                if isinstance(v, (list, tuple)) and len(v) > idx:
                    cand = str(v[idx]); break
                if isinstance(v, str):
                    cand = v; break
        elif isinstance(meta, (list, tuple)) and len(meta) > idx and isinstance(meta[idx], str):
            cand = meta[idx]
        elif isinstance(meta, str):
            cand = meta
    except Exception:
        cand = None
    if not cand:
        return ""
    name = Path(str(cand).replace("\\", "/")).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:60] or ""


class SavePredsEveryNStepsCallback(Callback):
    """
    Save yhat (and optionally triads) every N global steps during training.

    Files:
        {run_dir}/logs/pred_dumps/step_{S}_b{B}_i{I}[_<slug>]_yhat.wav
    CSV:
        {run_dir}/logs/pred_dumps/predictions.csv

    Parameters
    ----------
    out_subdir : str
        Subdirectory under run_dir/logs to write files.
    every_steps : int
        Save every N global steps (trainer.state.global_step).
    per_batch : int
        Max items to save from each triggering batch.
    max_total : int
        Stop saving after this many items (keeps logs small).
    save_triads : bool
        If True, save noisy/clean/yhat + residual (LS by default via save_wav_triads).
        If False (default), save yhat only (fastest, smallest).
    resid_mode : str
        Only used when save_triads=True. "noise" (noisy - alpha*clean) or "error" (clean - yhat).
    """

    def __init__(self,
                 out_subdir: str = "logs/pred_dumps",
                 every_steps: int = 50,
                 per_batch: int = 1,
                 max_total: int = 100,
                 save_triads: bool = False,
                 resid_mode: str = "noise"):
        self.out_subdir = str(out_subdir)
        self.every = int(every_steps)
        self.per_batch = int(per_batch)
        self.max_total = int(max_total)
        self.save_triads = bool(save_triads)
        self.resid_mode = str(resid_mode).lower()

        self._root: Optional[Path] = None
        self._sr: Optional[int] = None
        self._csv_path: Optional[Path] = None
        self._saved = 0

    # ---------- lifecycle ----------

    def on_fit_begin(self, trainer):
        # run_dir is normally set by your scripts; fall back to runs/default if missing
        run_dir = Path(getattr(trainer, "run_dir", "runs/default"))
        self._root = run_dir / self.out_subdir
        self._root.mkdir(parents=True, exist_ok=True)
        self._sr = infer_sr(trainer) or 48000
        self._csv_path = self._root / "predictions.csv"
        # ensure CSV header exists
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "step", "epoch", "batch_idx", "idx",
                    "base",
                    "yhat_rms_db", "yhat_silence_frac",
                    "snr_noisy_clean_db", "si_yhat_clean_db", "delta_si_db"
                ])
                w.writeheader()

    def on_train_batch_end(self, trainer, batch_idx: int,
                           batch: Dict[str, torch.Tensor] | List[torch.Tensor] | tuple,
                           outputs: Dict[str, torch.Tensor]) -> None:
        if self._root is None or self._sr is None or self._csv_path is None:
            return
        if self._saved >= self.max_total:
            return
        step = getattr(trainer.state, "global_step", 0)
        epoch = getattr(trainer.state, "epoch", 0)
        if self.every <= 0 or (step % self.every) != 0:
            return

        # unpack batch (dict or list/tuple)
        if isinstance(batch, dict):
            noisy = batch.get("noisy", None)
            clean = batch.get("clean", None)
            meta = batch.get("meta", None)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]
            meta = batch[2] if len(batch) >= 3 else None
        else:
            return

        yhat = outputs.get("yhat", None)
        if yhat is None or clean is None:
            return

        # batch size
        if yhat.dim() == 3:   # (B,1,T) or (B,C,T)
            B = yhat.size(0)
        elif yhat.dim() == 2: # (B,T)
            B = yhat.size(0)
        elif yhat.dim() == 1: # (T,)
            B = 1
            yhat = yhat.unsqueeze(0)
            clean = clean.unsqueeze(0) if clean.dim() == 1 else clean
            if noisy is not None and noisy.dim() == 1:
                noisy = noisy.unsqueeze(0)
        else:
            return

        n_to_save = min(self.per_batch, self.max_total - self._saved, B)
        for i in range(n_to_save):
            # slice per item, bring to (C,T) for saving
            def _to_ct(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if x is None: return None
                t = x[i]
                if t.dim() == 1: t = t.unsqueeze(0)
                if t.dim() == 3: t = t[0]
                return t.detach().to(torch.float32).cpu()

            y_ct = _to_ct(yhat)
            c_ct = _to_ct(clean)
            n_ct = _to_ct(noisy)

            # align lengths for metrics/saving
            if y_ct is not None and c_ct is not None:
                y_ct, c_ct = match_len(y_ct, c_ct)
            if y_ct is not None and n_ct is not None:
                y_ct, n_ct = match_len(y_ct, n_ct)

            # slug & base name
            slug = _slug_from_meta(meta, i) if meta is not None else ""
            base = f"step_{step:06d}_b{batch_idx:04d}_i{i}"
            if slug:
                base = f"{base}_{slug}"

            # save yhat (fast path)
            out_path = self._root / f"{base}_yhat.wav"
            save_wav_triads(
                self._root, base,
                Triad(clean=c_ct, noisy=None, yhat=y_ct, sr=self._sr),
                want_resid=False
            ) if self.save_triads is False else save_wav_triads(
                self._root, base,
                Triad(clean=c_ct, noisy=n_ct, yhat=y_ct, sr=self._sr),
                want_resid=True,  # use utils’ LS residual by default
            )

            # CSV: floats only
            row = {
                "step": int(step), "epoch": int(epoch), "batch_idx": int(batch_idx), "idx": int(i),
                "base": base,
                "yhat_rms_db": _as_float(rms_db(y_ct)) if y_ct is not None else float("nan"),
                "yhat_silence_frac": _as_float(silence_fraction(y_ct)) if y_ct is not None else float("nan"),
            }
            # triad metrics if we have noisy/clean
            if c_ct is not None:
                m = triad_metrics(y_ct if y_ct is not None else c_ct, c_ct, n_ct)
                for k, v in m.items():
                    row[k] = _as_float(v)

            with self._csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)

            self._saved += 1
