# -*- coding: utf-8 -*-
# tools/learning_from_audio_debug.py
"""
Aggregate metrics across audio_debug triads per epoch and report trends.

Adds:
- percent residual energy (100*||yhat-clean||^2 / ||noisy-clean||^2)
- composite score (0..100)
- trend slopes vs epoch
"""

from __future__ import annotations
import argparse, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from soundrestorer.utils.io import read_wav, write_csv
from soundrestorer.utils.audio import ensure_3d, match_length
from soundrestorer.utils.metrics import si_sdr_db, mae
from soundrestorer.utils.signal import stft_complex

def _find_epoch(p: Path) -> Optional[int]:
    m = re.search(r"(?:^|[_\-\/])ep(\d{1,3})(?:[_\-\/]|$)", str(p).replace("\\","/"))
    return int(m.group(1)) if m else None

def _gather_triads(root: Path) -> List[Dict]:
    wavs = list(root.rglob("*.wav"))
    groups: Dict[str, Dict[str, Path]] = {}
    for p in wavs:
        n = p.name.lower()
        stem, kind = n, None
        if n.endswith("_clean.wav"): stem = n[:-10]; kind = "clean"
        elif n.endswith("_noisy.wav"): stem = n[:-10]; kind = "noisy"
        elif n.endswith("_yhat.wav"):  stem = n[:-9];  kind = "yhat"
        if kind is None: continue
        key = f"{p.parent}/{stem}"
        groups.setdefault(key, {})[kind] = p
    triads = []
    for k, d in groups.items():
        if "clean" in d and "noisy" in d and "yhat" in d:
            triads.append({"base": k, **d})
    return triads

def _percent_residual_energy(clean: torch.Tensor, noisy: torch.Tensor, yhat: torch.Tensor) -> Optional[float]:
    a3, b3 = match_length(ensure_3d(clean), ensure_3d(noisy))
    y3, _  = match_length(ensure_3d(yhat), a3)
    num = (y3 - a3).pow(2).mean().item()
    den = (b3 - a3).pow(2).mean().item()
    if den <= 1e-12:
        return None
    return float(100.0 * (num / den))

def _lsd_db(a: torch.Tensor, b: torch.Tensor, n_fft: int = 2048, hop: int = 512, eps: float = 1e-8) -> float:
    A = stft_complex(ensure_3d(a), n_fft=n_fft, hop_length=hop)
    B = stft_complex(ensure_3d(b), n_fft=n_fft, hop_length=hop)
    Am = (A.abs().clamp_min(eps)).log10().mul(20.0)
    Bm = (B.abs().clamp_min(eps)).log10().mul(20.0)
    d = (Am - Bm).pow(2).mean(dim=-2).clamp_min(0).sqrt().mean()
    return float(d.item())

def _sigmoid(x: float) -> float: return 1.0 / (1.0 + math.exp(-x))

def _composite(si_impr_db: float, resid_pct: Optional[float], lsd_impr_db: float) -> float:
    # scaled 0..100
    si_term = _sigmoid(si_impr_db / 3.0)
    rp_term = 0.5 if resid_pct is None else max(0.0, 1.0 - resid_pct / 100.0)
    lsd_term = _sigmoid(lsd_impr_db / 3.0)
    return 100.0 * (0.45 * si_term + 0.35 * lsd_term + 0.20 * rp_term)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    triads = _gather_triads(root)
    if not triads:
        print("No triads found.")
        return

    by_ep: Dict[int, List[Dict]] = {}
    triad_rows = []

    for t in triads:
        ep = _find_epoch(Path(t["clean"])) or -1
        clean = read_wav(t["clean"], sr=args.sr, mono=args.mono)
        noisy = read_wav(t["noisy"], sr=args.sr, mono=args.mono)
        yhat  = read_wav(t["yhat"],  sr=args.sr, mono=args.mono)
        clean, noisy = match_length(clean, noisy)
        clean, yhat  = match_length(clean, yhat)

        si_nc = float(si_sdr_db(noisy, clean)[0].item())
        si_yc = float(si_sdr_db(yhat,  clean)[0].item())
        d_si  = si_yc - si_nc
        lsd_nc = _lsd_db(noisy, clean)
        lsd_yc = _lsd_db(yhat,  clean)
        resid_pct = _percent_residual_energy(clean, noisy, yhat)
        comp = _composite(d_si, resid_pct, max(0.0, lsd_nc - lsd_yc))

        triad_rows.append(dict(
            epoch=ep,
            base=t["base"],
            mae=float(mae(ensure_3d(yhat), ensure_3d(clean)).item()),
            resid_pct=resid_pct if resid_pct is not None else float("nan"),
            si_yhat_clean_db=si_yc,
            delta_si_db=d_si,
            lsd_impr_db=(lsd_nc - lsd_yc),
            composite_0_100=comp,
        ))
        by_ep.setdefault(ep, []).append(triad_rows[-1])

    def _median(vals: List[float]) -> float:
        t = torch.tensor([v for v in vals if v == v])
        return float(t.median().item()) if t.numel() else float("nan")

    print(f"\nFound {len(triad_rows)} triads across {len(by_ep)} epochs.")
    epoch_rows = []
    for ep in sorted(by_ep.keys()):
        R = by_ep[ep]
        epoch_rows.append(dict(
            epoch=ep,
            count=len(R),
            mae_med=_median([r["mae"] for r in R]),
            resid_pct_med=_median([r["resid_pct"] for r in R]),
            si_med=_median([r["si_yhat_clean_db"] for r in R]),
            delta_si_med=_median([r["delta_si_db"] for r in R]),
            lsd_impr_med=_median([r["lsd_impr_db"] for r in R]),
            composite_med=_median([r["composite_0_100"] for r in R]),
        ))

    for e in epoch_rows:
        print(f"  ep{e['epoch']:03d} | n={e['count']:2d} | MAE_med={e['mae_med']:.5f} | "
              f"resid%_med={e['resid_pct_med']:.1f} | SI_med={e['si_med']:+.2f} dB | "
              f"ΔSI_med={e['delta_si_med']:+.2f} dB | LSD_impr_med={e['lsd_impr_med']:+.2f} dB | "
              f"comp_med={e['composite_med']:.1f}")

    # trend slopes (simple least squares)
    def _slope(ep_vals: List[Tuple[int, float]]) -> float:
        if len(ep_vals) < 2: return float("nan")
        xs = torch.tensor([e for e,_ in ep_vals], dtype=torch.float32)
        ys = torch.tensor([v for _,v in ep_vals], dtype=torch.float32)
        x = xs - xs.mean()
        y = ys - ys.mean()
        return float((x*y).sum() / (x*x).sum().clamp_min(1e-8))

    if len(epoch_rows) >= 2:
        print("\nTrend slopes (per epoch):")
        for key, nice in [
            ("delta_si_med", "ΔSI"),
            ("resid_pct_med", "Residual%"),
            ("lsd_impr_med", "LSD_impr"),
            ("composite_med", "Composite")
        ]:
            s = _slope([(e["epoch"], e[key]) for e in epoch_rows if math.isfinite(e[key])])
            unit = "dB/ep" if "SI" in nice or "LSD" in nice else "%/ep" if "Residual" in nice else "/ep"
            sign_hint = "(+ better)" if nice in ("ΔSI","LSD_impr","Composite") else "(- better)"
            print(f"  {nice} slope: {s:+.3f} {unit} {sign_hint}")

    if args.write_csv:
        out1 = root / "learning_triad_metrics.csv"
        out2 = root / "learning_summary.csv"
        write_csv(out1, triad_rows, fieldnames=list(triad_rows[0].keys()))
        write_csv(out2, epoch_rows, fieldnames=list(epoch_rows[0].keys()))
        print(f"\nWrote {out1}")
        print(f"Wrote {out2}")

if __name__ == "__main__":
    main()
