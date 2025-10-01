# -*- coding: utf-8 -*-
# soundrestorer/eval/evaluator.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import csv, math, torch, statistics as stats
from pathlib import Path
from ..metrics.registry import compute_metrics
from ..metrics.common import rms_db


def _to_float_list(x: torch.Tensor) -> List[float]:
    x = x.detach().cpu().flatten()
    return [float(v) for v in x]


def robust_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return dict(mean=float("nan"), median=float("nan"), p25=float("nan"),
                    p75=float("nan"), min=float("nan"), max=float("nan"))
    vs = sorted(values)
    n = len(vs)
    q1 = vs[max(0, (n*25)//100)]
    q3 = vs[min(n-1, (n*75)//100)]
    return dict(
        mean = sum(vs)/n,
        median = stats.median(vs),
        p25 = q1,
        p75 = q3,
        min = vs[0],
        max = vs[-1],
    )


class Evaluator:
    """
    Accumulates per-item metrics; emits per-metric robust summaries.
    Use:
        ev = Evaluator(sr=48000, gate_db=-60.0)
        ev.add_triplet(yhat, clean, noisy, uid="ep010_idx5")
        ev.finalize(write_csv=".../eval.csv")
    """
    def __init__(self, sr: int = 48000, min_clean_rms_db: float = -80.0):
        self.sr = int(sr)
        self.min_clean_rms_db = float(min_clean_rms_db)
        self.rows: List[Dict[str, float]] = []
        self.keys: List[str] = []

    def add_triplet(self,
                    yhat: torch.Tensor,
                    clean: torch.Tensor,
                    noisy: Optional[torch.Tensor] = None,
                    uid: str = ""):
        # Silence gating on clean
        if rms_db(clean) < self.min_clean_rms_db:
            return

        with torch.no_grad():
            m = compute_metrics(yhat, clean, noisy, sr=self.sr, set_name="default_block")
            # Flatten tensors -> floats (per-item average if batched)
            flat: Dict[str, float] = {}
            for k, v in m.items():
                if torch.is_tensor(v):
                    flat[k] = float(v.mean().detach().cpu().item())
                elif isinstance(v, (int, float)):
                    flat[k] = float(v)
            flat["uid"] = uid
            self.rows.append(flat)
            for k in flat.keys():
                if k not in self.keys:
                    self.keys.append(k)

    def finalize(self,
                 write_csv: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Dict[str, float]]:
        # Aggregate per metric
        per_metric: Dict[str, List[float]] = {}
        for row in self.rows:
            for k, v in row.items():
                if k == "uid": continue
                per_metric.setdefault(k, []).append(float(v))

        summaries: Dict[str, Dict[str, float]] = {}
        for k, vals in per_metric.items():
            summaries[k] = robust_summary(vals)

        if verbose:
            print("==== EVALUATION SUMMARY (robust) ====")
            order = [
                "si_sdr_db", "delta_si_db", "si_err_ratio",
                "mrstft", "lsd_db", "mae", "rmse",
                "snr_noisy_db", "stoi", "pesq"
            ]
            for k in order:
                if k in summaries:
                    s = summaries[k]
                    if k in ("si_err_ratio", "mrstft", "lsd_db", "mae", "rmse"):
                        trend = "↓→0"
                    else:
                        trend = "↑ better"
                    print(f"{k:14s} | median={s['median']:.4f} | mean={s['mean']:.4f} | IQR=[{s['p25']:.4f},{s['p75']:.4f}] | {trend}")

        if write_csv:
            path = Path(write_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.keys)
                w.writeheader()
                for r in self.rows:
                    w.writerow(r)

        return summaries
