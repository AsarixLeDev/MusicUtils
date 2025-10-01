# -*- coding: utf-8 -*-
# tools/eval_manifest_pairs.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import torchaudio
from soundrestorer.eval.evaluator import Evaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str, help="JSONL with keys: clean, yhat [, noisy]")
    ap.add_argument("--sr", type=int, default=None, help="Optional resample rate (e.g., 48000)")
    ap.add_argument("--min-clean-rms-db", type=float, default=-80.0, help="Gate silent clips")
    ap.add_argument("--write-csv", type=str, default="", help="Where to write per-file CSV")
    args = ap.parse_args()

    ev = Evaluator(sr=args.sr or 48000, min_clean_rms_db=args.min_clean_rms_db)
    n = 0
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            clean_p = Path(row["clean"]); yhat_p = Path(row["yhat"])
            noisy_p = Path(row.get("noisy", "")) if "noisy" in row else None

            clean, sr_c = torchaudio.load(str(clean_p))
            yhat,  sr_y = torchaudio.load(str(yhat_p))
            if noisy_p is not None and noisy_p.exists():
                noisy, sr_n = torchaudio.load(str(noisy_p))
            else:
                noisy, sr_n = None, None

            sr = sr_c
            if args.sr: sr = args.sr
            if sr_y != sr: yhat  = torchaudio.functional.resample(yhat,  sr_y, sr)
            if sr_c != sr: clean = torchaudio.functional.resample(clean, sr_c, sr)
            if noisy is not None and sr_n != sr:
                noisy = torchaudio.functional.resample(noisy, sr_n, sr)

            uid = row.get("uid", f"item_{n}")
            ev.add_triplet(yhat, clean, noisy=noisy, uid=uid)
            n += 1

    csv_path = args.write_csf if args.write_csv else None
    ev.finalize(write_csv=args.write_csv or None, verbose=True)
    print(f"Evaluated {n} pairs")

if __name__ == "__main__":
    main()
