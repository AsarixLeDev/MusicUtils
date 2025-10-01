# -*- coding: utf-8 -*-
# tools/triad_diagnostics.py
"""
Diagnose triads in an audio_debug or data_audit directory.

Now includes:
- Log-Spectral Distance (LSD, dB)
- Percent residual energy (optional if resid.wav present)
- Composite score (0..100)
- Per-epoch trend summary

Triad expectations (flexible):
  <name>_clean.wav
  <name>_noisy.wav
  <name>_yhat.wav
  (optional) <name>_resid.wav  where resid = noisy - yhat  OR  clean - yhat depending on pipeline

Grouping handles nested epXXX/ folders and gathers epoch index automatically if present.

CSV:
  --write-csv  -> triad_diagnostics.csv  (per-item)
  --summary-csv -> triad_diagnostics_summary.csv (per-epoch medians)
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import torch

from soundrestorer.utils.io import read_wav, write_csv
from soundrestorer.utils.audio import ensure_3d, match_length, to_mono
from soundrestorer.utils.metrics import si_sdr_db, snr_db, mae
from soundrestorer.utils.signal import stft_complex

def _find_epoch(p: Path) -> Optional[int]:
    m = re.search(r"(?:^|[_\-\/])ep(\d{1,3})(?:[_\-\/]|$)", str(p).replace("\\","/"))
    return int(m.group(1)) if m else None

def _lsd_db(a: torch.Tensor, b: torch.Tensor, n_fft: int = 2048, hop: int = 512, eps: float = 1e-8) -> float:
    """
    Log-Spectral Distance in dB (lower is better).
    LSD = mean_t sqrt( mean_f ( (20*log10|A| - 20*log10|B|)^2 ) )
    """
    a3, b3 = ensure_3d(a), ensure_3d(b)
    a3, b3 = match_length(a3, b3)
    A = stft_complex(a3, n_fft=n_fft, hop_length=hop)  # [B,C,F,T]
    B = stft_complex(b3, n_fft=n_fft, hop_length=hop)
    Am = (A.abs().clamp_min(eps)).log10().mul(20.0)    # dB
    Bm = (B.abs().clamp_min(eps)).log10().mul(20.0)
    d = (Am - Bm) ** 2   # [B,C,F,T]
    d = d.mean(dim=-2)   # mean over F -> [B,C,T]
    d = d.clamp_min(0).sqrt()  # sqrt of mean square -> [B,C,T]
    # average over channels and time and batch
    return float(d.mean().item())

def _energy(x: torch.Tensor) -> float:
    x3 = ensure_3d(x)
    return float((x3 ** 2).mean().item())

def _percent_residual_energy(clean: torch.Tensor, noisy: torch.Tensor, yhat: torch.Tensor) -> Optional[float]:
    """
    % residual energy vs noisy error baseline:
      100 * ||yhat-clean||^2 / ||noisy-clean||^2
    Returns None if denominator ~0.
    """
    a3, b3 = match_length(ensure_3d(clean), ensure_3d(noisy))
    y3, _  = match_length(ensure_3d(yhat), a3)
    num = (y3 - a3).pow(2).mean().item()
    den = (b3 - a3).pow(2).mean().item()
    if den <= 1e-12:
        return None
    return float(100.0 * (num / den))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _composite_score(
    si_improve_db: float,
    lsd_noisy_db: float,
    lsd_yhat_db: float,
    resid_pct: Optional[float],
) -> float:
    """
    0..100 score; higher is better.
    - Reward ΔSI dB
    - Reward LSD reduction
    - Reward low residual % energy
    Tuned to be forgiving; adjust to taste.
    """
    # Normalize SI improvement: ~+6 dB => near 1.0
    si_term = _sigmoid(si_improve_db / 3.0)

    # Normalize LSD improvement: (noisy - yhat) / 3 dB
    lsd_impr = max(0.0, lsd_noisy_db - lsd_yhat_db)
    lsd_term = _sigmoid(lsd_impr / 3.0)

    # Residual energy term: 0% -> 1.0 ; 100% -> ~0
    if resid_pct is None:
        resid_term = 0.5
    else:
        resid_term = max(0.0, 1.0 - (resid_pct / 100.0))
        resid_term = min(1.0, resid_term)

    score01 = 0.45 * si_term + 0.35 * lsd_term + 0.20 * resid_term
    return 100.0 * score01

def _gather_triads(root: Path) -> List[Dict]:
    wavs = list(root.rglob("*.wav"))
    groups: Dict[str, Dict[str, Path]] = {}
    for p in wavs:
        n = p.name.lower()
        stem = n
        kind = None
        if n.endswith("_clean.wav"):
            stem = n[:-10]; kind = "clean"
        elif n.endswith("_noisy.wav"):
            stem = n[:-10]; kind = "noisy"
        elif n.endswith("_yhat.wav"):
            stem = n[:-9];  kind = "yhat"
        elif n.endswith("_resid.wav"):
            stem = n[:-10]; kind = "resid"
        else:
            # allow alternative labels
            if "_clean" in n: stem = n.split("_clean")[0]; kind = "clean"
            elif "_noisy" in n: stem = n.split("_noisy")[0]; kind = "noisy"
            elif "_yhat" in n: stem = n.split("_yhat")[0]; kind = "yhat"
            elif "_resid" in n: stem = n.split("_resid")[0]; kind = "resid"
        if kind is None:
            # ignore non-triad wavs
            continue
        key = f"{p.parent}/{stem}"
        if key not in groups: groups[key] = {}
        groups[key][kind] = p

    triads = []
    for k, d in groups.items():
        if "clean" in d and "noisy" in d and "yhat" in d:
            triads.append({"base": k, **d})
    return triads

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="audio_debug or data_audit directory")
    ap.add_argument("--sr", type=int, default=None, help="optional resample")
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--summary-csv", action="store_true")
    ap.add_argument("--mono", action="store_true", help="force mono before metrics")
    ap.add_argument("--fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.root)
    triads = _gather_triads(root)
    if not triads:
        print("No triads found.")
        return

    rows = []
    by_epoch: Dict[int, List[Dict]] = {}

    for t in triads:
        ep = _find_epoch(Path(t["clean"])) or _find_epoch(Path(t["noisy"])) or _find_epoch(Path(t["yhat"])) or -1
        clean = read_wav(t["clean"], sr=args.sr, mono=args.mono)
        noisy = read_wav(t["noisy"], sr=args.sr, mono=args.mono)
        yhat  = read_wav(t["yhat"],  sr=args.sr, mono=args.mono)

        # align
        clean, noisy = match_length(clean, noisy)
        clean, yhat  = match_length(clean, yhat)

        # Core metrics
        si_noisy = float(si_sdr_db(noisy, clean)[0].item())
        si_yhat  = float(si_sdr_db(yhat,  clean)[0].item())
        d_si     = si_yhat - si_noisy

        snr_nc   = float(snr_db(noisy, clean)[0].item())
        snr_yc   = float(snr_db(yhat,  clean)[0].item())

        lsd_nc = _lsd_db(noisy, clean, n_fft=args.fft, hop=args.hop)
        lsd_yc = _lsd_db(yhat,  clean, n_fft=args.fft, hop=args.hop)

        resid_pct = _percent_residual_energy(clean, noisy, yhat)

        comp = _composite_score(d_si, lsd_nc, lsd_yc, resid_pct)

        rows.append(dict(
            epoch=ep,
            base=t["base"],
            si_noisy_db=si_noisy,
            si_yhat_db=si_yhat,
            delta_si_db=d_si,
            snr_noisy_clean_db=snr_nc,
            snr_yhat_clean_db=snr_yc,
            lsd_noisy_clean_db=lsd_nc,
            lsd_yhat_clean_db=lsd_yc,
            lsd_impr_db=(lsd_nc - lsd_yc),
            resid_pct=resid_pct if resid_pct is not None else float("nan"),
            mae_yhat_clean=float(mae(ensure_3d(yhat), ensure_3d(clean)).item()),
            composite_0_100=comp,
        ))
        by_epoch.setdefault(ep, []).append(rows[-1])

    print(f"\nAnalyzed {len(rows)} triads.")
    # global quick stats
    si_imprs = torch.tensor([r["delta_si_db"] for r in rows if math.isfinite(r["delta_si_db"])])
    lsd_impr = torch.tensor([r["lsd_impr_db"] for r in rows if math.isfinite(r["lsd_impr_db"])])
    resid_ok = torch.tensor([r["resid_pct"] for r in rows if math.isfinite(r["resid_pct"])]) if any(math.isfinite(r["resid_pct"]) for r in rows) else None

    def _med(t: torch.Tensor) -> float: return float(t.median().item()) if t.numel() else float("nan")

    print(f" ΔSI dB: median={_med(si_imprs):+.2f} | mean={float(si_imprs.mean().item()):+.2f}")
    print(f" LSD improvement: median={_med(lsd_impr):+.2f} dB")
    if resid_ok is not None and resid_ok.numel():
        print(f" Residual energy: median={_med(resid_ok):.1f}%")

    # Per-epoch medians and simple trend slopes
    def _slope(ep_vals: List[Tuple[int, float]]) -> float:
        # simple least squares slope
        if len(ep_vals) < 2: return float("nan")
        xs = torch.tensor([e for e,_ in ep_vals], dtype=torch.float32)
        ys = torch.tensor([v for _,v in ep_vals], dtype=torch.float32)
        x = xs - xs.mean()
        y = ys - ys.mean()
        denom = (x*x).sum().clamp_min(1e-8)
        return float((x*y).sum() / denom)

    print("\nPer-epoch summary (medians):")
    epoch_rows = []
    for ep in sorted(by_epoch.keys()):
        R = by_epoch[ep]
        med = lambda k: float(torch.tensor(
            [r[k] for r in R if r[k] == r[k]]  # filter NaNs
        ).median().item()) if R else float("nan")

        epoch_rows.append(dict(
            epoch=ep,
            count=len(R),
            delta_si_db_med=med("delta_si_db"),
            lsd_impr_db_med=med("lsd_impr_db"),
            resid_pct_med=med("resid_pct"),
            composite_med=med("composite_0_100"),
        ))
        print(f"  ep{ep:03d} | n={len(R):2d} | ΔSI_med={epoch_rows[-1]['delta_si_db_med']:+.2f} dB | "
              f"LSD_impr_med={epoch_rows[-1]['lsd_impr_db_med']:+.2f} dB | "
              f"resid_med={epoch_rows[-1]['resid_pct_med']:.1f}% | comp_med={epoch_rows[-1]['composite_med']:.1f}")

    # trends
    if len(epoch_rows) >= 2:
        dsi_slope = _slope([(e["epoch"], e["delta_si_db_med"]) for e in epoch_rows if math.isfinite(e["delta_si_db_med"])])
        lsd_slope = _slope([(e["epoch"], e["lsd_impr_db_med"]) for e in epoch_rows if math.isfinite(e["lsd_impr_db_med"])])
        rp_slope  = _slope([(e["epoch"], e["resid_pct_med"]) for e in epoch_rows if math.isfinite(e["resid_pct_med"])])
        comp_slope= _slope([(e["epoch"], e["composite_med"]) for e in epoch_rows if math.isfinite(e["composite_med"])])
        print("\nTrend slopes per epoch (approx; + is better for ΔSI/LSD_impr/comp; - is better for resid_pct):")
        print(f"  ΔSI slope: {dsi_slope:+.3f} dB/epoch")
        print(f"  LSD_impr slope: {lsd_slope:+.3f} dB/epoch")
        print(f"  Residual% slope: {rp_slope:+.3f} %/epoch")
        print(f"  Composite slope: {comp_slope:+.3f} /epoch")

    if args.write_csv:
        out = root / "triad_diagnostics.csv"
        write_csv(out, rows, fieldnames=list(rows[0].keys()))
        print(f"\nWrote {out}")

    if args.summary_csv and epoch_rows:
        out = root / "triad_diagnostics_summary.csv"
        write_csv(out, epoch_rows, fieldnames=list(epoch_rows[0].keys()))
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
