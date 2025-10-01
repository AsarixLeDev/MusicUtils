#!/usr/bin/env python3
"""
Analyze audio_debug WAV triads (clean/noisy/yhat) over epochs and compute a learning curve
using only the WAV content.

Per-triad metrics:
  - MAE / MSE between yhat and clean
  - Residual energy ratio (||yhat-clean||^2 / ||clean||^2) in dB  -> “goes toward -inf if perfect”
  - SI-SDR(yhat, clean) in dB
  - If noisy exists: SI-SDR(noisy, clean) and ΔSI = SI-SDR(yhat, clean) - SI-SDR(noisy, clean)
  - Silence fractions, RMS/peak dBFS (for sanity)

Aggregates per epoch (median/mean). Writes CSV if requested.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio


def load_mono(path: Path) -> Tuple[torch.Tensor, int]:
    """Load WAV -> mono float32 1D tensor in [-1,1], return (wave, sr)."""
    wav, sr = torchaudio.load(str(path))  # (C,T)
    wav = wav.to(torch.float32)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=False)
    elif wav.dim() == 2:
        wav = wav[0]
    elif wav.dim() == 1:
        pass
    else:
        raise RuntimeError(f"Unexpected shape from {path}: {tuple(wav.shape)}")
    return wav.contiguous(), int(sr)


def align_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim both to same length (min)."""
    n = min(a.numel(), b.numel())
    return a[:n], b[:n]


def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> float:
    rms = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return float(20.0 * torch.log10(rms + eps))


def peak_dbfs(x: torch.Tensor, eps: float = 1e-12) -> float:
    peak = torch.max(torch.abs(x.float()))
    return float(20.0 * torch.log10(peak + eps))


def silence_frac(x: torch.Tensor, thr_dbfs: float = -60.0) -> float:
    thr = 10.0 ** (thr_dbfs / 20.0)
    return float((torch.abs(x) < thr).float().mean())


def snr_db(clean: torch.Tensor, noisy: torch.Tensor, eps: float = 1e-12) -> float:
    """SNR(noisy,clean) = 10*log10(||clean||^2 / ||noisy-clean||^2)"""
    c, y = align_pair(clean, noisy)
    num = torch.sum(c ** 2).clamp_min(eps)
    den = torch.sum((y - c) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(num / den))


def sisdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8, match_length: bool = True) -> float:
    """Scale-invariant SDR for 1D tensors (batchless)."""
    if match_length:
        y, x = align_pair(y, x)
    y = y - y.mean()
    x = x - x.mean()
    s = (torch.sum(y * x) / (torch.sum(x * x) + eps)) * x
    e = y - s
    num = torch.sum(s ** 2)
    den = torch.sum(e ** 2) + eps
    return float(10.0 * torch.log10(num / den + eps))


def resid_ratio_db(clean: torch.Tensor, yhat: torch.Tensor, eps: float = 1e-12) -> float:
    """Residual energy ratio in dB: 10*log10(||yhat-clean||^2 / ||clean||^2). Tends to -inf if perfect."""
    c, y = align_pair(clean, yhat)
    num = torch.sum((y - c) ** 2).clamp_min(eps)
    den = torch.sum(c ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(num / den))


def find_triad_roots(root: Path) -> List[Tuple[int, Path]]:
    """
    Find triad roots and their epochs. A root has *_clean.wav and *_yhat.wav (noisy optional).
    We extract epoch from 'epXXX_' prefix anywhere in the filename.
    Returns list of (epoch, basepath_without_suffix).
    """
    wavs = list(root.rglob("*.wav"))
    groups: Dict[str, Dict[str, Path]] = {}
    epoch_of: Dict[str, int] = {}

    ep_re = re.compile(r"ep(\d+)_", re.IGNORECASE)

    for p in wavs:
        name = p.name.lower()
        suf = None
        if name.endswith("_clean.wav"): suf = "clean"
        elif name.endswith("_yhat.wav"): suf = "yhat"
        elif name.endswith("_noisy.wav"): suf = "noisy"
        if suf is None:
            continue
        base = str(p)[:-len(f"_{suf}.wav")]
        groups.setdefault(base, {})[suf] = p

        # try to parse epoch
        m = ep_re.search(name)
        if m:
            try:
                epoch = int(m.group(1))
            except Exception:
                epoch = -1
        else:
            epoch = -1
        epoch_of[base] = epoch

    triads = []
    for base, d in groups.items():
        if "clean" in d and "yhat" in d:  # need at least these two
            triads.append((epoch_of.get(base, -1), Path(base)))
    triads.sort(key=lambda t: (t[0], str(t[1])))
    return triads


def analyze_triad(base: Path, silence_dbfs: float, match_length: bool) -> Dict[str, object]:
    paths = {
        "clean": Path(str(base) + "_clean.wav"),
        "yhat":  Path(str(base) + "_yhat.wav"),
        "noisy": Path(str(base) + "_noisy.wav"),
    }
    rec: Dict[str, object] = {"id": str(base)}

    have = {k: p.exists() for k, p in paths.items()}
    if not have["clean"] or not have["yhat"]:
        rec["ok"] = False
        rec["reason"] = "missing clean or yhat"
        return rec
    rec["ok"] = True

    # load
    c, sr_c = load_mono(paths["clean"])
    y, sr_y = load_mono(paths["yhat"])
    if sr_c != sr_y:
        # Skip resampling: for audio_debug this shouldn't happen; trim after alignment
        pass

    # basic stats
    r = {}
    r["sr"] = sr_c
    r["dur_sec"] = round(min(c.numel(), y.numel()) / sr_c, 4)
    r["clean_rms_dbfs"] = round(rms_dbfs(c), 2)
    r["yhat_rms_dbfs"]  = round(rms_dbfs(y), 2)
    r["clean_peak_dbfs"] = round(peak_dbfs(c), 2)
    r["yhat_peak_dbfs"]  = round(peak_dbfs(y), 2)
    r["clean_silence_frac"] = round(silence_frac(c, silence_dbfs), 4)
    r["yhat_silence_frac"]  = round(silence_frac(y, silence_dbfs), 4)

    # error metrics
    c0, y0 = align_pair(c, y)
    mae = torch.mean(torch.abs(y0 - c0)).item()
    mse = torch.mean((y0 - c0) ** 2).item()
    r["mae"] = float(mae)
    r["mse"] = float(mse)
    r["resid_ratio_db"] = round(resid_ratio_db(c, y), 2)
    r["sisdr_yhat_clean_db"] = round(sisdr_db(y, c, match_length=match_length), 2)

    # if we have noisy, compute SNR + ΔSI
    if have["noisy"]:
        n, sr_n = load_mono(paths["noisy"])
        r["snr_noisy_clean_db"] = round(snr_db(c, n), 2)
        si_n = sisdr_db(n, c, match_length=match_length)
        si_y = sisdr_db(y, c, match_length=match_length)
        r["sisdr_noisy_clean_db"] = round(si_n, 2)
        r["delta_sisdr_db"] = round(si_y - si_n, 2)

    rec.update(r)
    return rec


def summarize_epoch(rows: List[Dict[str, object]]) -> Dict[str, float]:
    """Compute robust aggregates for a list of triads in an epoch."""
    def col(name):
        vals = [float(r[name]) for r in rows if name in r and isinstance(r[name], (int, float))]
        return vals

    import statistics as st
    out = {}
    for k in [
        "mae", "mse", "resid_ratio_db",
        "sisdr_yhat_clean_db", "sisdr_noisy_clean_db", "delta_sisdr_db",
        "snr_noisy_clean_db",
    ]:
        vals = col(k)
        if vals:
            out[f"{k}_mean"] = float(st.mean(vals))
            out[f"{k}_median"] = float(st.median(vals))
    out["count"] = float(len(rows))
    return out


def main():
    ap = argparse.ArgumentParser("Measure learning from audio_debug WAVs")
    ap.add_argument("--root", required=True, help="Folder containing epXXX_idxYYY_(clean|noisy|yhat).wav (recursively)")
    ap.add_argument("--silence-dbfs", type=float, default=-60.0, help="Silence threshold for fractions (default -60 dBFS)")
    ap.add_argument("--match-length", type=str, default="true", help="Trim pairs to min length (true/false)")
    ap.add_argument("--write-csv", action="store_true", help="Write per-triad CSV and per-epoch summary CSV")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    match_length = str(args.match_length).lower() in ("1", "true", "yes", "y")

    triads = find_triad_roots(root)
    if not triads:
        print(f"No triads found under {root}. Need *_clean.wav and *_yhat.wav; *_noisy.wav optional.")
        return

    # Analyze
    all_rows: List[Dict[str, object]] = []
    by_epoch: Dict[int, List[Dict[str, object]]] = {}
    for ep, base in triads:
        rec = analyze_triad(base, args.silence_dbfs, match_length)
        rec["epoch"] = ep
        all_rows.append(rec)
        by_epoch.setdefault(ep, []).append(rec)

    # Print epoch summaries
    print(f"\nFound {len(all_rows)} triads across {len(by_epoch)} epochs.")
    print("Per-epoch summary (medians unless noted):")
    for ep in sorted(by_epoch.keys()):
        rows = [r for r in by_epoch[ep] if r.get("ok", False)]
        s = summarize_epoch(rows)
        msg = [f"ep{ep:03d}",
               f"count={int(s.get('count',0))}",
               f"MAE_med={s.get('mae_median','-'):.5f}" if "mae_median" in s else "",
               f"resid_dB_med={s.get('resid_ratio_db_median','-')}",
               f"SI(yhat,clean)_med={s.get('sisdr_yhat_clean_db_median','-')}",
              ]
        if "delta_sisdr_db_median" in s:
            msg.append(f"ΔSI_med={s['delta_sisdr_db_median']:+.2f} dB")
        print("  " + " | ".join([m for m in msg if m]))

    # Write CSVs (optional)
    if args.write_csv:
        # per-triad
        out1 = root / "learning_triad_metrics.csv"
        keys = sorted({k for r in all_rows for k in r.keys()})
        with open(out1, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"Wrote {out1}")

        # per-epoch summary
        out2 = root / "learning_summary.csv"
        sum_rows = []
        sum_keys = set(["epoch"])
        for ep in sorted(by_epoch.keys()):
            s = summarize_epoch([r for r in by_epoch[ep] if r.get("ok", False)])
            s["epoch"] = ep
            sum_rows.append(s)
            sum_keys.update(s.keys())
        sum_keys = sorted(sum_keys)
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sum_keys)
            w.writeheader()
            for s in sum_rows:
                w.writerow(s)
        print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
