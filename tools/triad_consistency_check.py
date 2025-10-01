# -*- coding: utf-8 -*-
# tools/triad_consistency_check.py
from __future__ import annotations
import argparse, re, csv, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torchaudio

from soundrestorer.metrics.common import (
    to_mono, rms_db, si_sdr_db, rel_error_db,
    silence_fraction, clip_fraction, resample_if_needed
)

@dataclass
class TriadRow:
    epoch: str
    base: str
    resid_as: str                   # "clean_minus_yhat" / "noisy_minus_yhat" / "unknown"
    err_db_identity: float          # relative error in dB for the chosen identity
    si_yhat_clean_db: float
    si_noisy_clean_db: Optional[float]
    delta_si_db: Optional[float]
    clean_rms_db: float
    yhat_rms_db: float
    noisy_rms_db: Optional[float]
    resid_rms_db: Optional[float]
    clean_silence: float            # [0..1]
    yhat_silence: float
    noisy_silence: Optional[float]
    resid_silence: Optional[float]
    clean_clip: float               # [0..1]
    yhat_clip: float
    noisy_clip: Optional[float]
    resid_clip: Optional[float]

TRIAD_PATTERNS = {
    "clean": re.compile(r"_clean\.wav$", re.IGNORECASE),
    "yhat":  re.compile(r"_yhat\.wav$",  re.IGNORECASE),
    "noisy": re.compile(r"_noisy\.wav$", re.IGNORECASE),
    "resid": re.compile(r"_resid\.wav$", re.IGNORECASE),
}

def _group_triads(root: Path) -> Dict[str, List[Dict[str, Path]]]:
    out = {}
    for wav in root.rglob("*.wav"):
        tag = wav.parent.name  # epoch dir (e.g. ep010)
        name = wav.name
        base = (name
                .replace("_clean.wav","")
                .replace("_yhat.wav","")
                .replace("_noisy.wav","")
                .replace("_resid.wav",""))
        out.setdefault(tag, {})
        out[tag].setdefault(base, {})
        for k, pat in TRIAD_PATTERNS.items():
            if pat.search(name):
                out[tag][base][k] = wav
    grouped = {tag: [ {"base": b, **paths} for b, paths in d.items() if "clean" in paths and "yhat" in paths] for tag, d in out.items()}
    return grouped

def _load(p: Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(p))
    return wav, sr

def _median(xs: List[float]) -> float:
    xs = sorted([x for x in xs if x is not None and not math.isnan(x)])
    return xs[len(xs)//2] if xs else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="audio_debug or data_audit root")
    ap.add_argument("--sr", type=int, default=None, help="Optional resample rate (e.g. 48000)")
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    triads = _group_triads(root)
    if not triads:
        print(f"[triad-consistency] No triads found under {root}")
        return

    rows: List[TriadRow] = []

    for ep, items in sorted(triads.items()):
        for it in items:
            clean, sr_c = _load(it["clean"])
            yhat,  sr_y = _load(it["yhat"])
            noisy = resid = None
            sr_n = sr_r = None
            if "noisy" in it: noisy, sr_n = _load(it["noisy"])
            if "resid" in it: resid, sr_r = _load(it["resid"])

            # unify sr if requested
            srx = sr_c if args.sr is None else args.sr
            clean = resample_if_needed(clean, sr_c, srx)
            yhat  = resample_if_needed(yhat,  sr_y, srx)
            if noisy is not None: noisy = resample_if_needed(noisy, sr_n, srx)
            if resid is not None: resid = resample_if_needed(resid, sr_r, srx)

            # Decide resid semantics
            resid_as = "unknown"; err_identity = float("nan")
            if resid is not None:
                errA = rel_error_db(clean, yhat + resid) if clean is not None else +1e6
                errB = rel_error_db(noisy, yhat + resid) if noisy is not None else +1e6
                if errA < errB:
                    resid_as, err_identity = "clean_minus_yhat", errA
                else:
                    resid_as, err_identity = "noisy_minus_yhat", errB

            si_yc = si_sdr_db(yhat, clean)
            si_nc = si_sdr_db(noisy, clean) if noisy is not None else None
            dsi = (si_yc - si_nc) if si_nc is not None else None

            row = TriadRow(
                epoch=ep, base=it["base"], resid_as=resid_as, err_db_identity=err_identity,
                si_yhat_clean_db=si_yc, si_noisy_clean_db=si_nc, delta_si_db=dsi,
                clean_rms_db=rms_db(clean), yhat_rms_db=rms_db(yhat),
                noisy_rms_db=(rms_db(noisy) if noisy is not None else None),
                resid_rms_db=(rms_db(resid) if resid is not None else None),
                clean_silence=silence_fraction(clean), yhat_silence=silence_fraction(yhat),
                noisy_silence=(silence_fraction(noisy) if noisy is not None else None),
                resid_silence=(silence_fraction(resid) if resid is not None else None),
                clean_clip=clip_fraction(clean), yhat_clip=clip_fraction(yhat),
                noisy_clip=(clip_fraction(noisy) if noisy is not None else None),
                resid_clip=(clip_fraction(resid) if resid is not None else None)
            )
            rows.append(row)

    # summary
    resid_modes = {"clean_minus_yhat":0, "noisy_minus_yhat":0, "unknown":0}
    for r in rows: resid_modes[r.resid_as] = resid_modes.get(r.resid_as, 0) + 1

    print("\n==== TRIAD CONSISTENCY SUMMARY ====")
    print(f"triads: {len(rows)} | resid semantics counts: {resid_modes}")
    print(f"identity err(dB) median: {_median([r.err_db_identity for r in rows]):.2f} (≤ -40dB is tight)")
    print(f"ΔSI median: {_median([r.delta_si_db for r in rows]):.2f} dB")
    print(f"clean silence>95%: {sum(r.clean_silence>0.95 for r in rows)}/{len(rows)}")
    print(f" yhat silence>95%: {sum(r.yhat_silence>0.95 for r in rows)}/{len(rows)}")

    if args.write_csv:
        csv_path = (Path(args.root) / "triad_consistency.csv")
        keys = list(asdict(rows[0]).keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(asdict(r))
        print(f"Wrote {csv_path}")

    print("Done.")

if __name__ == "__main__":
    main()
