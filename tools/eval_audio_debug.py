# -*- coding: utf-8 -*-
# tools/eval_audio_debug.py
from __future__ import annotations
import argparse, re
from pathlib import Path
import torchaudio, torch
from soundrestorer.eval.evaluator import Evaluator

TRIAD_PATTERNS = {
    "clean": re.compile(r"_clean\.wav$", re.IGNORECASE),
    "yhat":  re.compile(r"_yhat\.wav$",  re.IGNORECASE),
    "noisy": re.compile(r"_noisy\.wav$", re.IGNORECASE),
    "resid": re.compile(r"_resid\.wav$", re.IGNORECASE),
}

def group_triads(root: Path):
    """
    Return dict: { epoch_tag: [ {base, clean, yhat, noisy, resid}, ... ] }
    Works for audio_debug or data_audit layout.
    """
    out = {}
    for wav in root.rglob("*.wav"):
        name = wav.name
        epoch_dir = wav.parent
        # epoch dir usually like ep010/ or similar
        tag = epoch_dir.name
        # derive base id: strip suffixes
        base = (name
                .replace("_clean.wav","")
                .replace("_yhat.wav","")
                .replace("_noisy.wav","")
                .replace("_resid.wav",""))
        key = (tag, base)
        out.setdefault(tag, {})
        out[tag].setdefault(base, {})
        for k, pat in TRIAD_PATTERNS.items():
            if pat.search(name):
                out[tag][base][k] = wav
    # flatten to lists
    grouped = {tag: [ {"base": b, **paths} for b, paths in d.items() if "clean" in paths and "yhat" in paths] for tag, d in out.items()}
    return grouped

def load_wav(p: Path):
    wav, sr = torchaudio.load(str(p))
    return wav, sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="Path to audio_debug or data_audit root")
    ap.add_argument("--sr", type=int, default=None, help="Optional resample rate (e.g., 48000)")
    ap.add_argument("--min-clean-rms-db", type=float, default=-80.0, help="Gate: drop clips whose clean RMS dB below this")
    ap.add_argument("--write-csv", action="store_true", help="Write per-file CSV next to root")
    args = ap.parse_args()

    root = Path(args.root)
    groups = group_triads(root)
    if not groups:
        print(f"No triads found under {root}")
        return

    for epoch_tag, items in sorted(groups.items()):
        ev = Evaluator(sr=args.sr or None or 48000, min_clean_rms_db=args.min_clean_rms_db)
        count = 0
        for it in items:
            clean_p = it.get("clean"); yhat_p = it.get("yhat")
            noisy_p = it.get("noisy", None)
            clean, sr_c = load_wav(clean_p)
            yhat,  sr_y = load_wav(yhat_p)
            noisy, sr_n = (load_wav(noisy_p) if noisy_p is not None else (None, None))

            # ensure common sr
            sr = sr_c
            if args.sr: sr = args.sr
            if sr_y != sr: yhat = torchaudio.functional.resample(yhat, sr_y, sr)
            if sr_c != sr: clean = torchaudio.functional.resample(clean, sr_c, sr)
            if noisy is not None and sr_n != sr: noisy = torchaudio.functional.resample(noisy, sr_n, sr)

            ev.add_triplet(yhat, clean, noisy=noisy, uid=f"{epoch_tag}/{it['base']}")
            count += 1

        print(f"\nEpoch {epoch_tag}: evaluated {count} triads")
        csv_path = None
        if args.write_csv:
            csv_path = str((root / f"eval_{epoch_tag}.csv").resolve())
        ev.finalize(write_csv=csv_path, verbose=True)

if __name__ == "__main__":
    main()
