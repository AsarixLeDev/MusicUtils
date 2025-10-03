# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch

from .callbacks import Callback
from .utils import Triad, save_wav_triads, infer_sr
from ..metrics.common import match_len


class SavePredsEveryNStepsCallback(Callback):
    """
    Save model predictions every N global steps.
    Compatible with batches as dict ({"noisy","clean"}) or tuple/list (noisy, clean, ...).
    Accepts outputs["yhat"] with shape (B,T) or (B,1,T) or (B,C,T); saved as mono.

    Args:
      out_subdir: subdir under run_dir to write files
      every_steps: trigger period
      per_batch: how many items from the triggering batch to save
      max_total: safety cap for the whole run
      save_triads: if True, also save _noisy/_clean/_resid next to _yhat
      save_csv: if True, append rows to preds.csv (ep, gs, base, paths)
      dtype: 'float32' or 'pcm16' for WAV samples
    """
    def __init__(self,
                 out_subdir: str = "logs/pred_dumps",
                 every_steps: int = 50,
                 per_batch: int = 1,
                 max_total: int = 50,
                 save_triads: bool = True,
                 save_csv: bool = True,
                 dtype: str = "float32", **_):
        self.out_subdir = out_subdir
        self.every = int(every_steps)
        self.per_batch = int(per_batch)
        self.max_total = int(max_total)
        self.save_triads = bool(save_triads)
        self.save_csv = bool(save_csv)
        self.dtype = str(dtype).lower()

        self._root: Optional[Path] = None
        self._sr: Optional[int] = None
        self._total_saved: int = 0
        self._csv_rows: List[Dict[str, Any]] = []

    # ----- helpers -----
    @staticmethod
    def _to_bt(x: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T); (B,1,T)->(B,T); (B,C,T)->mono (B,T)
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if x.size(1) == 1:
                return x[:, 0, :]
            return x.mean(dim=1)
        raise RuntimeError(f"Unexpected yhat shape {tuple(x.shape)}")

    @staticmethod
    def _pick_item(t: Optional[torch.Tensor], i: int) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if t.dim() == 3:      # (B,C,T)
            return t[i].detach().to(torch.float32).cpu()
        if t.dim() == 2:      # (B,T) treat as mono -> (1,T)
            return t[i].unsqueeze(0).detach().to(torch.float32).cpu()
        if t.dim() == 1:      # (T,) -> (1,T)
            return t.unsqueeze(0).detach().to(torch.float32).cpu()
        raise RuntimeError(f"Unexpected tensor shape {tuple(t.shape)}")

    def _base_name(self, epoch: int, gstep: int, i: int, outputs: Dict[str, Any]) -> str:
        extra = ""
        ids = outputs.get("ids", None)
        if isinstance(ids, torch.Tensor) and ids.numel() > i:
            try:
                extra = f"_id{int(ids.view(-1)[i].item())}"
            except Exception:
                extra = ""
        return f"ep{epoch:03d}_gs{gstep:06d}_b{i:02d}{extra}"

    # ----- lifecycle -----
    def on_fit_begin(self, trainer=None, **_):
        run_dir = Path(getattr(trainer, "run_dir", "runs/default"))
        self._root = run_dir / self.out_subdir
        self._root.mkdir(parents=True, exist_ok=True)
        self._sr = infer_sr(trainer) or 48000
        self._total_saved = 0
        self._csv_rows = []
        print("[save-preds] ready:"
              f" out={self._root} | every={self.every} | per_batch={self.per_batch} | max_total={self.max_total}")

    def on_batch_end(self, trainer=None, state=None, batch=None, outputs=None, **_):
        if self._root is None or self._sr is None:
            return
        if self._total_saved >= self.max_total:
            return
        if not isinstance(outputs, dict):
            return

        gs = int(getattr(state, "global_step", 0) or 0)
        ep = int(getattr(state, "epoch", 0) or 0)
        if gs <= 0 or (gs % self.every) != 0:
            return

        # get yhat
        yhat = outputs.get("yhat", None)
        if yhat is None:
            print("[save-preds] skip: outputs['yhat'] missing")
            return

        # normalize yhat -> (B,T) CPU float32
        try:
            ybt = self._to_bt(yhat.detach().to(torch.float32)).cpu().contiguous()
        except Exception as e:
            print(f"[save-preds] skip: bad yhat shape {tuple(getattr(yhat,'shape',()))}: {e}")
            return

        # access clean/noisy from batch for triads (best effort)
        clean = noisy = None
        if isinstance(batch, dict):
            noisy = batch.get("noisy", None)
            clean = batch.get("clean", None)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]

        B = ybt.shape[0]
        take = min(self.per_batch, B, self.max_total - self._total_saved)

        import torchaudio
        rows_now = []

        for i in range(take):
            base = self._base_name(ep, gs, i, outputs)
            y_i = ybt[i].unsqueeze(0)  # (1,T)

            # Save only yhat fast path if triads disabled
            if not self.save_triads:
                out_p = self._root / f"{base}_yhat.wav"
                try:
                    if self.dtype == "pcm16":
                        torchaudio.save(str(out_p), y_i, self._sr, encoding="PCM_S", bits_per_sample=16)
                    else:
                        torchaudio.save(str(out_p), y_i, self._sr, encoding="PCM_F", bits_per_sample=32)
                    rows_now.append(dict(epoch=ep, gstep=gs, base=base, yhat=str(out_p)))
                except Exception as e:
                    print(f"[save-preds] write failed: {e}")
                continue

            # Triad save
            c_i = self._pick_item(clean, i)
            n_i = self._pick_item(noisy, i)

            # make lengths match where possible
            if c_i is not None:
                y_i, c_i = match_len(y_i, c_i)
            if n_i is not None:
                y_i, n_i = match_len(y_i, n_i)

            tri = Triad(clean=c_i, noisy=n_i, yhat=y_i, sr=self._sr)
            try:
                paths = save_wav_triads(self._root, base, tri, want_resid=True)
                rows_now.append(dict(epoch=ep, gstep=gs, base=base, **{k: str(v) for k, v in paths.items()}))
            except Exception as e:
                print(f"[save-preds] triad save failed: {e}")

        if rows_now:
            self._csv_rows.extend(rows_now)
            self._total_saved += len(rows_now)
            print(f"[save-preds] gs={gs} saved {len(rows_now)} (total {self._total_saved}) -> {self._root}")

    def on_epoch_end(self, trainer=None, state=None, **_):
        if not self.save_csv or not self._csv_rows or self._root is None:
            return
        csv_path = self._root / "preds.csv"
        keys = sorted({k for r in self._csv_rows for k in r.keys()})
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if f.tell() == 0:
                w.writeheader()
            for r in self._csv_rows:
                w.writerow(r)
        print(f"[save-preds] wrote {csv_path}")
        self._csv_rows.clear()
