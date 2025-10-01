# -*- coding: utf-8 -*-
# tools/audit_stems.py
"""
Verify that stems sum (approximately) to mixture.

Supports:
  A) Directory with per-track subfolders containing stems + mixture
  B) JSONL manifest with fields:
     - "mixture": path
     - "stems": list of paths  (or dict of {name: path})

Report per-track relative error in dB: 20*log10(||mix - sum|| / ||mix||)
Lower (more negative) is better. Typical tolerance ~ -40 dB or lower.

Usage examples:
  python tools/audit_stems.py --root /data/MUSDB18HQ --tol-db -40
  python tools/audit_stems.py --manifest stems.jsonl --tol-db -40
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from soundrestorer.utils.io import read_wav, read_jsonl, write_csv
from soundrestorer.utils.audio import ensure_3d, match_length, to_mono, normalize_peak

def _rel_err_db(sum_: torch.Tensor, mix: torch.Tensor, eps: float = 1e-12) -> float:
    sum_, mix = match_length(ensure_3d(sum_), ensure_3d(mix))
    e = (sum_ - mix)
    num = float((e**2).mean().item()) + eps
    den = float((mix**2).mean().item()) + eps
    return 10.0 * torch.log10(torch.tensor(num / den)).item()

def _from_manifest(manifest: Path) -> List[Dict]:
    rows = read_jsonl(manifest)
    out = []
    for r in rows:
        if "mixture" in r and "stems" in r:
            stems = r["stems"]
            if isinstance(stems, dict): stems = list(stems.values())
            out.append({"id": r.get("id", str(len(out))), "mixture": r["mixture"], "stems": stems})
    return out

def _from_dir(root: Path, exts={".wav", ".flac"}) -> List[Dict]:
    """
    Expect per-track subfolders with one mixture file and N stems.
    Heuristics: treat any file with 'mix' or 'mixture' in name as mixture; else sum stems.
    """
    tracks = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        wavs = [p for p in d.iterdir() if p.suffix.lower() in exts]
        if not wavs: continue
        mixture = None
        stems = []
        for w in wavs:
            n = w.name.lower()
            if "mix" in n or "mixture" in n or n.startswith("mixture"):
                mixture = w
            else:
                stems.append(w)
        if mixture is None or not stems:
            # fallback: if exactly 5 wavs use first as mixture
            if len(wavs) >= 2:
                mixture = wavs[0]
                stems = wavs[1:]
            else:
                continue
        tracks.append({"id": d.name, "mixture": str(mixture), "stems": [str(s) for s in stems]})
    return tracks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--root", type=str, default=None, help="root folder with tracks")
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--tol-db", type=float, default=-40.0, help="pass if rel err <= tol")
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    items: List[Dict]
    if args.manifest:
        items = _from_manifest(Path(args.manifest))
    elif args.root:
        items = _from_dir(Path(args.root))
    else:
        raise SystemExit("Please provide --manifest or --root")

    if not items:
        print("No tracks found.")
        return

    rows = []
    n_pass = 0
    for it in items:
        mix = read_wav(it["mixture"], sr=args.sr, mono=args.mono)
        mix = normalize_peak(mix, target=0.98)
        acc = torch.zeros_like(ensure_3d(mix))
        for sp in it["stems"]:
            s = read_wav(sp, sr=args.sr, mono=args.mono)
            s, _ = match_length(s, mix)
            acc = acc + s
        err_db = _rel_err_db(acc, mix)
        ok = err_db <= args.tol_db
        if ok: n_pass += 1
        rows.append(dict(id=it["id"], mixture=it["mixture"], stems=len(it["stems"]), rel_err_db=err_db, pass_=bool(ok)))
        print(f"[{ 'OK' if ok else 'FAIL' }] {it['id']}: rel_err {err_db:+.2f} dB (tol {args.tol_db:+.1f})")

    print(f"\nSummary: {n_pass}/{len(items)} passed ({100.0*n_pass/len(items):.1f}%).")
    if args.write_csv:
        out = Path(args.root or ".") / "audit_stems.csv"
        write_csv(out, rows, fieldnames=list(rows[0].keys()))
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
