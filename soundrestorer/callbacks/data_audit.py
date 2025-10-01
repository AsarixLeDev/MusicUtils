# -*- coding: utf-8 -*-
from __future__ import annotations

import csv, math
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

from soundrestorer.callbacks.utils import (
    Triad, save_wav_triads, triad_metrics, infer_sr
)
from soundrestorer.metrics.common import (
    silence_fraction, match_len
)

from soundrestorer.utils.metrics import rms_db

def _as_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return float("nan")


class DataAuditCallback:
    """
    Save a small, representative subset of train batches in each epoch and
    emit a CSV `audit.csv` with quick stats (SNR, SI, silence %, RMS in dB).

    Files go to:
        {run_dir}/logs/data_audit/ep{epoch:03d}/...
    and a CSV at:
        {epoch_dir}/audit.csv
    """

    def __init__(
        self,
        *,
        max_items: int = 12,
        take_from_batches: int = 6,
        subdir: str = "logs/data_audit",
        sr: Optional[int] = None,
        silence_threshold: float = 0.95,    # fraction
        write_wavs: bool = True,
        write_csv: bool = True,
    ):
        self.max_items = int(max_items)
        self.take_from_batches = int(take_from_batches)
        self.subdir = subdir
        self.force_sr = sr
        self.silence_threshold = float(silence_threshold)
        self.write_wavs = bool(write_wavs)
        self.write_csv = bool(write_csv)

        self._epoch = 0
        self._saved = 0
        self._rows: List[Dict[str, float]] = []
        self._ep_dir: Optional[Path] = None

    def on_fit_begin(self, trainer) -> None:
        if self.force_sr is None:
            self.force_sr = infer_sr(trainer)

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        self._epoch = int(epoch)
        self._saved = 0
        self._rows = []
        run_dir = Path(getattr(trainer, "run_dir", "runs/default"))
        self._ep_dir = run_dir / self.subdir / f"ep{epoch:03d}"
        print(f"[data-audit] will save up to {self.max_items} items "
              f"from {self.take_from_batches} batches -> {self._ep_dir}")

    def on_train_batch_end(
        self,
        trainer,
        batch_idx: int,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        if self._saved >= self.max_items: return
        if batch_idx >= self.take_from_batches: return
        if self._ep_dir is None: return

        yhat = outputs.get("yhat", None)
        clean = batch.get("clean", None)
        noisy = batch.get("noisy", None)
        if yhat is None or clean is None:
            return

        B = yhat.shape[0] if yhat.dim() == 3 else 1
        # save up to ~max_items/take_from_batches items from this batch
        per_batch = max(1, math.ceil(self.max_items / self.take_from_batches))
        count_here = min(per_batch, self.max_items - self._saved)

        for i in range(min(B, count_here)):
            y_i = yhat[i:i+1]
            c_i = clean[i:i+1]
            n_i = noisy[i:i+1] if noisy is not None else None

            # squash batch dim to [C,T]
            if y_i.dim() == 3: y_i = y_i[0]
            if c_i.dim() == 3: c_i = c_i[0]
            if n_i is not None and n_i.dim() == 3: n_i = n_i[0]

            # align lengths
            y_i, c_i = match_len(y_i, c_i)
            if n_i is not None:
                y_i, n_i = match_len(y_i, n_i)

            base = f"b{batch_idx}_i{i}"
            sr = self.force_sr or infer_sr(trainer) or 48000

            if self.write_wavs:
                save_wav_triads(self._ep_dir, base, Triad(c_i, n_i, y_i, sr),
                                want_resid=True, prefer_clean_resid=True)

            # record floats only
            r = dict(
                base=base,
                clean_rms_db=_as_float(rms_db(c_i)),
                yhat_rms_db=_as_float(rms_db(y_i)),
                clean_silence_frac=_as_float(silence_fraction(c_i)),
                yhat_silence_frac=_as_float(silence_fraction(y_i)),
            )
            # add triad metrics (already floats after our previous patch, but _as_float keeps it safe)
            m = triad_metrics(y_i, c_i, n_i)
            for k, v in m.items():
                r[k] = _as_float(v)

            if n_i is not None:
                r["noisy_rms_db"] = _as_float(rms_db(n_i))
                r["noisy_silence_frac"] = _as_float(silence_fraction(n_i))

            self._rows.append(r)
            self._saved += 1

    def on_epoch_end(self, trainer, epoch: int) -> None:
        if not self._rows or not self.write_csv:
            return

        csv_path = self._ep_dir / "audit.csv"
        # write per-item CSV
        with csv_path.open("w", newline="") as f:
            keys = sorted(self._rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in self._rows: w.writerow(r)

        # human summary (floats only)
        snrs = [float(r.get("snr_noisy_clean_db")) for r in self._rows
                if r.get("snr_noisy_clean_db") is not None]
        if snrs:
            mean_snr = float(sum(snrs) / len(snrs))
            min_snr  = float(min(snrs))
            max_snr  = float(max(snrs))
        else:
            mean_snr = min_snr = max_snr = float("nan")

        nsil = [float(r.get("noisy_silence_frac", 0.0)) for r in self._rows
                if r.get("noisy_silence_frac") is not None]
        n_sil95 = sum(1 for v in nsil if v > self.silence_threshold)

        print(f"[data-audit] saved {len(self._rows)} items | "
              f"SNR mean {mean_snr:+.2f} dB (min {min_snr:+.2f}, max {max_snr:+.2f}) | "
              f"noisy_sil>{int(self.silence_threshold*100)}% count {n_sil95}")
        print(f"[data-audit] wrote {csv_path}")
