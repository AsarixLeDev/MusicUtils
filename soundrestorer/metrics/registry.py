# -*- coding: utf-8 -*-
# soundrestorer/metrics/registry.py
from __future__ import annotations

from typing import Dict, Optional

import torch

from .common import (
    si_sdr_db, si_sdr_error_ratio, snr_db,
    lsd_db, mrstft_metric, stoi_avg, pesq_avg, to_mono, match_lengths
)


# Name -> callable signature: f(yhat, clean, noisy=None, sr=48000) -> Dict[str, Tensor/float]
def _metric_block(
        yhat: torch.Tensor,
        clean: torch.Tensor,
        noisy: Optional[torch.Tensor],
        sr: int
) -> Dict[str, torch.Tensor]:
    """
    Default comprehensive block of metrics.
    All positive 'loss-like' metrics trend to 0 when perfect:
      - si_err_ratio, mrstft, lsd_db, mae, rmse
    Human-friendly metrics:
      - si_sdr_db, delta_si_db, snr_noisy_db
      - stoi, pesq (if packages available)
    """
    yhat, clean = to_mono(yhat), to_mono(clean)
    yhat, clean = match_lengths(yhat, clean)
    out: Dict[str, torch.Tensor] = {}

    # Waveform MAE / RMSE
    diff = (yhat - clean)
    out["mae"] = diff.abs().mean(dim=-1)
    out["rmse"] = torch.sqrt((diff ** 2).mean(dim=-1))

    # SI-SDR (dB) and error ratio
    si_db = si_sdr_db(yhat, clean)
    err_ratio = si_sdr_error_ratio(yhat, clean)
    out["si_sdr_db"] = si_db
    out["si_err_ratio"] = err_ratio

    # ΔSI vs noisy (if provided)
    if noisy is not None:
        noisy = to_mono(noisy)
        noisy, _ = match_lengths(noisy, clean)
        si_n = si_sdr_db(noisy, clean)
        out["delta_si_db"] = si_db - si_n
        out["snr_noisy_db"] = snr_db(noisy, clean)

    # Spectral metrics
    out["lsd_db"] = lsd_db(yhat, clean)  # RMS dB distance (→0)
    out["mrstft"] = mrstft_metric(yhat, clean)  # →0

    # Optional STOI/PESQ if available and sr supported
    stoi = stoi_avg(yhat, clean, sr)
    if stoi is not None: out["stoi"] = stoi
    pesq = pesq_avg(yhat, clean, sr)
    if pesq is not None: out["pesq"] = pesq

    return out


METRIC_REGISTRY = {
    "default_block": _metric_block,
}


def compute_metrics(
        yhat: torch.Tensor,
        clean: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        sr: int = 48000,
        set_name: str = "default_block"
) -> Dict[str, torch.Tensor]:
    if set_name not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric set '{set_name}'. Choose from {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[set_name](yhat, clean, noisy, sr)
