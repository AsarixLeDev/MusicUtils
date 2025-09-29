#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio


def load_mono(path: Path) -> Tuple[torch.Tensor, int]:
    """Load WAV -> mono float32 in [-1,1], shape (T,), return (wave, sr)."""
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


def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> float:
    """RMS in dBFS (0 dBFS is full-scale sine ~ 0.707 peak); for quick diagnostics we use 20*log10(rms)."""
    rms = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return float(20.0 * torch.log10(rms + eps))


def peak_dbfs(x: torch.Tensor, eps: float = 1e-12) -> float:
    peak = torch.max(torch.abs(x.float()))
    return float(20.0 * torch.log10(peak + eps))


def silence_ratio(
    x: torch.Tensor, thresh_dbfs: float = -60.0
) -> float:
    """
    Fraction of samples below |thresh|. Simple and fast.
    For a robust frame-based measure you could compute short-term RMS, but this catches common issues.
    """
    thr = 10.0 ** (thresh_dbfs / 20.0)
    frac = float((torch.abs(x) < thr).float().mean())
    return frac


def align_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim both to same length (min)."""
    n = min(a.numel(), b.numel())
    return a[:n], b[:n]


def snr_db(ref_clean: torch.Tensor, test_noisy: torch.Tensor, eps: float = 1e-12) -> float:
    """
    SNR between clean and noisy: 10*log10(||clean||^2 / ||noisy-clean||^2).
    Assumes aligned and same length.
    """
    x, y = align_pair(ref_clean, test_noisy)
    num = torch.sum(x ** 2).clamp_min(eps)
    den = torch.sum((y - x) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(num / den))


def si_sdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8, match_length=True) -> float:
    """
    Scale-Invariant SDR per batchless pair (1D). Same math as your robust function but scalar here.
    If match_length=True, both are trimmed to min length.
    """
    if match_length:
        y, x = align_pair(y, x)
    y = y - y.mean()
    x = x - x.mean()
    s = (torch.sum(y * x) / (torch.sum(x * x) + eps)) * x
    e = y - s
    num = torch.sum(s ** 2)
    den = torch.sum(e ** 2) + eps
    return float(10.0 * torch.log10(num / den + eps))


def find_triad_roots(root: Path) -> List[Path]:
    """
    Find "roots" that have *_clean.wav and *_noisy.wav; *_yhat.wav optional.
    A root is full path minus the suffix _clean/_noisy/_yhat.
    """
    wavs = list(root.rglob("*.wav"))
    # map from base -> types found
    groups: Dict[str, Dict[str, Path]] = {}
    for p in wavs:
        name = p.name.lower()
        for tag in ("_clean.wav", "_noisy.wav", "_yhat.wav"):
            if name.endswith(tag):
                base = str(p)[:-len(tag)]  # remove suffix
                g = groups.setdefault(base, {})
                g[tag[1:-4]] = p  # clean/noisy/yhat
                break
    triad_bases = []
    for base, d in groups.items():
        if "clean" in d and "noisy" in d:
            triad_bases.append(Path(base))
    triad_bases.sort()
    return triad_bases


def analyze_triad(base: Path, match_length: bool, silence_dbfs: float) -> Dict[str, object]:
    paths = {
        "clean": Path(str(base) + "_clean.wav"),
        "noisy": Path(str(base) + "_noisy.wav"),
        "yhat":  Path(str(base) + "_yhat.wav"),
    }
    rec: Dict[str, object] = {"id": str(base)}

    # Load files that exist
    waves: Dict[str, torch.Tensor] = {}
    srs: Dict[str, int] = {}
    for k, p in paths.items():
        if p.exists():
            w, sr = load_mono(p)
            waves[k] = w
            srs[k] = sr
            rec[f"{k}_sr"] = sr
            rec[f"{k}_dur_sec"] = round(w.numel() / sr, 4)
            rec[f"{k}_rms_dbfs"] = round(rms_dbfs(w), 2)
            rec[f"{k}_peak_dbfs"] = round(peak_dbfs(w), 2)
            rec[f"{k}_silence_frac"] = round(silence_ratio(w, silence_dbfs), 4)

    # Pairwise metrics (require clean+noisy)
    if "clean" in waves and "noisy" in waves:
        clean, noisy = waves["clean"], waves["noisy"]
        if match_length:
            clean, noisy = align_pair(clean, noisy)
        rec["snr_noisy_vs_clean_db"] = round(snr_db(clean, noisy), 2)
        rec["sisdr_noisy_clean_db"] = round(si_sdr_db(noisy, clean, match_length=False), 2)

        # True residual noise (what was added)
        resid_true = noisy - clean
        rec["resid_true_rms_dbfs"] = round(rms_dbfs(resid_true), 2)
        rec["resid_true_peak_dbfs"] = round(peak_dbfs(resid_true), 2)
        rec["resid_true_silence_frac"] = round(silence_ratio(resid_true, silence_dbfs), 4)

    # Predicted metrics (require yhat+clean)
    if "clean" in waves and "yhat" in waves:
        clean, yhat = waves["clean"], waves["yhat"]
        if match_length:
            clean, yhat = align_pair(clean, yhat)
        rec["sisdr_yhat_clean_db"] = round(si_sdr_db(yhat, clean, match_length=False), 2)

        # ΔSI = improvement over noisy
        if "noisy" in waves:
            noisy = waves["noisy"]
            if match_length:
                noisy, _ = align_pair(noisy, clean)
            delta = si_sdr_db(yhat, clean, match_length=False) - si_sdr_db(noisy, clean, match_length=False)
            rec["delta_sisdr_db"] = round(delta, 2)

        # Predicted residual (what the model leaves vs clean)
        resid_pred = yhat - clean
        rec["resid_pred_rms_dbfs"] = round(rms_dbfs(resid_pred), 2)
        rec["resid_pred_peak_dbfs"] = round(peak_dbfs(resid_pred), 2)
        rec["resid_pred_silence_frac"] = round(silence_ratio(resid_pred, silence_dbfs), 4)

        # How similar is predicted residual to true residual? (if noisy present)
        if "noisy" in waves:
            resid_true = align_pair(waves["noisy"], waves["clean"])[0] - align_pair(waves["noisy"], waves["clean"])[1]
            resid_pred = align_pair(waves["yhat"], waves["clean"])[0] - align_pair(waves["yhat"], waves["clean"])[1]
            n = min(resid_true.numel(), resid_pred.numel())
            rt, rp = resid_true[:n], resid_pred[:n]
            # SI-SDR between residuals: high means model leaves noise similar to the true noise (not desired)
            rec["sisdr_resid_pred_vs_true_db"] = round(si_sdr_db(rp, rt, match_length=False), 2)

    return rec


def summarize(recs: List[Dict[str, object]]) -> None:
    import statistics as stats

    def col(name):
        vals = [float(r[name]) for r in recs if name in r and isinstance(r[name], (int, float))]
        return vals

    N = len(recs)
    print(f"\nAnalyzed {N} triads.")
    for name in [
        "snr_noisy_vs_clean_db",
        "sisdr_noisy_clean_db",
        "sisdr_yhat_clean_db",
        "delta_sisdr_db",
    ]:
        vals = col(name)
        if vals:
            print(f"  {name:>26}: mean={stats.mean(vals):+6.2f} dB | median={stats.median(vals):+6.2f} dB "
                  f"| min={min(vals):+6.2f} | max={max(vals):+6.2f}")

    # Silence diagnostics
    for field in ["noisy_silence_frac", "clean_silence_frac", "yhat_silence_frac"]:
        vals = col(field)
        if vals:
            hi = sum(v > 0.95 for v in vals)
            print(f"  {field:>26}: >95% silence in {hi}/{len(vals)} files ({100.0*hi/len(vals):.1f}%)")

    # How often is "noisy" effectively too clean?
    s_noisy = col("snr_noisy_vs_clean_db")
    if s_noisy:
        over_25 = sum(v > 25.0 for v in s_noisy)
        print(f"  noisy vs clean SNR > 25 dB in {over_25}/{len(s_noisy)} items ({100.0*over_25/len(s_noisy):.1f}%)\n")


def main():
    ap = argparse.ArgumentParser("Diagnose noisy/clean/yhat WAV triads")
    ap.add_argument("--root", required=True, help="Folder containing *_noisy/_clean/_yhat wavs (recursively)")
    ap.add_argument("--write-csv", action="store_true", help="Write triad_diagnostics.csv next to --root")
    ap.add_argument("--silence-dbfs", type=float, default=-60.0, help="Silence threshold in dBFS (default -60)")
    ap.add_argument("--match-length", type=str, default="true",
                    help="Trim pairs to min length before metrics (true/false)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    match_length = str(args.match_length).lower() in ("1", "true", "yes", "y")

    bases = find_triad_roots(root)
    if not bases:
        print(f"No triads found under {root} (need *_clean.wav and *_noisy.wav; *_yhat.wav optional).")
        return

    recs: List[Dict[str, object]] = []
    for b in bases:
        try:
            rec = analyze_triad(b, match_length, args.silence_dbfs)
            recs.append(rec)
        except Exception as e:
            print(f"[warn] failed on {b}: {e}")

    summarize(recs)

    if args.write_csv:
        out = root / "triad_diagnostics.csv"
        # union of keys
        all_keys = []
        for r in recs:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in recs:
                w.writerow(r)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
