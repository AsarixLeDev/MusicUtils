# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch

# Be tolerant to where Callback is defined in your tree
try:
    from soundrestorer.callbacks.callbacks import Callback  # your newer location
except Exception:
    try:
        from soundrestorer.core.trainer import Callback  # legacy
    except Exception:  # fallback stub (tests/tools-only)
        class Callback:  # type: ignore
            pass

from soundrestorer.callbacks.utils import (
    Triad, save_wav_triads, triad_metrics, infer_sr
)
from soundrestorer.metrics.common import silence_fraction, match_len
from soundrestorer.utils.metrics import rms_db


def _as_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return float("nan")


def _slug_from_meta(meta: Any, idx: int) -> str:
    """
    Best-effort: derive a short, filesystem-safe slug from batch metadata.
    Tries (in order):
      meta['path'], meta['clean_path'], meta['noisy_path'], list/tuple entries, str(meta)
    """
    cand: Optional[str] = None

    def _safe(s: str) -> str:
        s = s.replace("\\", "/")
        name = Path(s).stem
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return name[:60] or ""

    try:
        if isinstance(meta, dict):
            for k in ("path", "clean_path", "noisy_path"):
                v = meta.get(k)
                if isinstance(v, (list, tuple)) and len(v) > idx:
                    cand = str(v[idx])
                    break
                if isinstance(v, str):
                    cand = v
                    break
        elif isinstance(meta, (list, tuple)) and len(meta) > idx:
            # meta is already a list of paths
            if isinstance(meta[idx], str):
                cand = meta[idx]
        elif isinstance(meta, str):
            cand = meta
    except Exception:
        cand = None

    return _safe(cand) if cand else ""


class DataAuditCallback(Callback):
    """
    Save a small, representative subset of *train* batches each epoch and
    emit a CSV `audit.csv` with quick stats (SNR, SI, silence %, RMS in dB).

    Files go to:
        {run_dir}/logs/data_audit/ep{epoch:03d}/...
    and a CSV at:
        {epoch_dir}/audit.csv

    NOTE:
      - Runs at on_train_batch_end (post-augment) to reflect what the model actually trained on.
      - Appends a slug (track/ID) to filenames when metadata is available.
    """

    def __init__(
        self,
        *,
        max_items: int = 12,
        take_from_batches: int = 6,
        subdir: str = "logs/data_audit",
        sr: Optional[int] = None,
        silence_threshold: float = 0.95,  # fraction
        write_wavs: bool = True,
        write_csv: bool = True,
    ):
        self.max_items = int(max_items)
        self.take_from_batches = int(take_from_batches)
        self.subdir = str(subdir)
        self.force_sr = None if sr is None else int(sr)
        self.silence_threshold = float(silence_threshold)
        self.write_wavs = bool(write_wavs)
        self.write_csv = bool(write_csv)

        self._epoch: int = 0
        self._saved: int = 0
        self._rows: List[Dict[str, float]] = []
        self._ep_dir: Optional[Path] = None
        self._batch_seen: int = 0

    # ---------- lifecycle ----------

    def on_fit_begin(self, trainer) -> None:
        if self.force_sr is None:
            self.force_sr = infer_sr(trainer)

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        self._epoch = int(epoch)
        self._saved = 0
        self._rows = []
        self._batch_seen = 0
        run_dir = Path(getattr(trainer, "run_dir", "runs/default"))
        self._ep_dir = run_dir / self.subdir / f"ep{self._epoch:03d}"
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[data-audit] will save up to {self.max_items} items from {self.take_from_batches} batches -> {self._ep_dir}"
        )

    def on_train_batch_end(
        self,
        trainer,
        batch_idx: int,
        batch: Dict[str, torch.Tensor] | List[torch.Tensor] | tuple,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        if self._ep_dir is None:
            return
        if self._saved >= self.max_items:
            return
        if batch_idx >= self.take_from_batches:
            return

        # Expect a dict batch in your trainer; be tolerant to list/tuple
        if isinstance(batch, dict):
            yhat = outputs.get("yhat", None)
            clean = batch.get("clean", None)
            noisy = batch.get("noisy", None)
            meta = batch.get("meta", None)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            yhat = outputs.get("yhat", None)
            noisy, clean = batch[0], batch[1]
            # meta might be present as third element (per-item); keep optional
            meta = batch[2] if len(batch) >= 3 else None
        else:
            return

        if yhat is None or clean is None:
            return

        # batch dims
        B = yhat.shape[0] if isinstance(yhat, torch.Tensor) and yhat.dim() >= 2 else 1

        # save up to ~max_items/take_from_batches items from this batch
        per_batch = max(1, math.ceil(self.max_items / self.take_from_batches))
        count_here = min(per_batch, self.max_items - self._saved)

        for i in range(min(B, count_here)):
            # slice per item
            def _pick(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if x is None:
                    return None
                if x.dim() >= 3:
                    return x[i].detach()
                if x.dim() == 2 and x.size(0) == B:
                    return x[i].detach()
                return x.detach()

            y_i = _pick(yhat)
            c_i = _pick(clean)
            n_i = _pick(noisy)

            if y_i is None or c_i is None:
                continue

            # shapes to [C,T]
            if y_i.dim() == 3: y_i = y_i[0]
            if c_i.dim() == 3: c_i = c_i[0]
            if n_i is not None and n_i.dim() == 3: n_i = n_i[0]

            # align lengths on T
            y_i, c_i = match_len(y_i, c_i)
            if n_i is not None:
                y_i, n_i = match_len(y_i, n_i)

            # slug from meta if present
            slug = _slug_from_meta(meta, i) if meta is not None else ""
            base = f"{slug}_b{batch_idx}_i{i}" if slug else f"b{batch_idx}_i{i}"

            sr = self.force_sr or infer_sr(trainer) or 48000

            if self.write_wavs:
                save_wav_triads(
                    self._ep_dir,
                    base,
                    Triad(clean=c_i, noisy=n_i, yhat=y_i, sr=sr),
                    want_resid=True,
                )

            # record floats only (CSV)
            r = dict(
                base=base,
                clean_rms_db=_as_float(rms_db(c_i)),
                yhat_rms_db=_as_float(rms_db(y_i)),
                clean_silence_frac=_as_float(silence_fraction(c_i)),
                yhat_silence_frac=_as_float(silence_fraction(y_i)),
            )
            # add triad metrics (already floats if utils.triad_metrics was patched; _as_float for belt-and-suspenders)
            m = triad_metrics(y_i, c_i, n_i)
            for k, v in m.items():
                r[k] = _as_float(v)

            if n_i is not None:
                r["noisy_rms_db"] = _as_float(rms_db(n_i))
                r["noisy_silence_frac"] = _as_float(silence_fraction(n_i))

            self._rows.append(r)
            self._saved += 1

    def on_epoch_end(self, trainer, epoch: int) -> None:
        if self._ep_dir is None:
            return
        if not self._rows or not self.write_csv:
            print(f"[data-audit] saved {self._saved} items | no CSV (write_csv={self.write_csv})")
            return

        csv_path = self._ep_dir / "audit.csv"
        with csv_path.open("w", newline="") as f:
            keys = sorted(self._rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self._rows:
                w.writerow(r)

        # human summary (floats only)
        snrs = [float(r.get("snr_noisy_clean_db")) for r in self._rows if r.get("snr_noisy_clean_db") is not None]
        if snrs:
            mean_snr = float(sum(snrs) / len(snrs))
            min_snr = float(min(snrs))
            max_snr = float(max(snrs))
        else:
            mean_snr = min_snr = max_snr = float("nan")

        nsil = [float(r.get("noisy_silence_frac", 0.0)) for r in self._rows if r.get("noisy_silence_frac") is not None]
        n_sil95 = sum(1 for v in nsil if v > self.silence_threshold)

        print(
            f"[data-audit] saved {self._saved} items | "
            f"SNR mean {mean_snr:+.2f} dB (min {min_snr:+.2f}, max {max_snr:+.2f}) | "
            f"noisy_sil>{int(self.silence_threshold * 100)}% count {n_sil95}"
        )
        print(f"[data-audit] wrote {csv_path}")
