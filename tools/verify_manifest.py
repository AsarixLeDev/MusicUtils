#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify JSONL manifests:
- Checks that audio files exist & are readable
- Collects sample_rate, duration, channel count
- (Optional) quick-RMS scan to flag near-silence
- Emits a concise CSV and a summary printout

Supports "clean" manifests (music mixtures) and "noise" manifests.
"""

import argparse, json, csv, sys, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torchaudio
import torch

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as ex:
                print(f"[WARN] bad JSON line in {path}: {ex}", file=sys.stderr)
    return rows

def rms(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    return float(torch.sqrt(torch.clamp((x**2).mean(), min=eps)))

def rms_db(x: torch.Tensor, eps: float = 1e-12) -> float:
    r = rms(x, eps)
    return 20.0 * math.log10(max(r, eps))

def quick_probe_audio(path: Path, max_frames: int = 48000*5) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Optional[int]]:
    """Return (sr, num_frames, channels, dur_s, rms_db, read_frames) or None on failure."""
    try:
        info = torchaudio.info(str(path))
        sr   = info.sample_rate
        nf   = info.num_frames
        ch   = info.num_channels
        dur  = nf / max(1, sr)
        # quick read up to max_frames
        to_read = nf if nf <= max_frames else max_frames
        wav, _  = torchaudio.load(str(path), frame_offset=0, num_frames=to_read)  # (C, T)
        if wav.numel() == 0:
            return sr, nf, ch, dur, None, 0
        # mono RMS (just to flag near-silence)
        if wav.size(0) > 1:
            wavm = wav.mean(0, keepdim=True)
        else:
            wavm = wav
        return sr, nf, ch, dur, rms_db(wavm), wav.size(-1)
    except Exception as ex:
        print(f"[ERR] reading {path}: {ex}", file=sys.stderr)
        return None, None, None, None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="Path to JSONL")
    ap.add_argument("--out_csv", type=str, default="", help="Write per-item CSV summary")
    ap.add_argument("--kind", type=str, choices=["clean", "noise", "auto"], default="auto",
                    help="'clean' expects key 'clean'; 'noise' expects key 'noise'")
    ap.add_argument("--silence_rms_db", type=float, default=-60.0,
                    help="Flag items below this RMS(dB) as near-silent (on the probed chunk)")
    ap.add_argument("--probe_max_s", type=float, default=5.0, help="Seconds to probe for RMS")
    args = ap.parse_args()

    man_path = Path(args.manifest)
    rows = load_jsonl(man_path)
    if not rows:
        print(f"[ERR] empty or unreadable manifest: {man_path}", file=sys.stderr)
        sys.exit(2)

    # infer key
    key = "clean" if args.kind in ("clean", "auto") and "clean" in rows[0] else None
    if key is None and (args.kind in ("noise", "auto")):
        key = "noise" if "noise" in rows[0] else None
    if key is None:
        print("[ERR] could not infer manifest kind; use --kind clean|noise", file=sys.stderr)
        sys.exit(3)

    max_frames = int(round(float(args.probe_max_s) * 48000))  # will be clipped by real SR at read time

    csv_rows = []
    exists_ok = 0
    read_ok   = 0
    near_sil  = 0
    sr_hist   = {}
    dur_total = 0.0

    for i, r in enumerate(rows):
        p = Path(r.get(key, ""))
        if not p.exists():
            csv_rows.append([i, str(p), 0, "", "", "", "", "MISSING"])
            continue
        exists_ok += 1
        sr, nf, ch, dur, rdb, read_frames = quick_probe_audio(p, max_frames=max_frames)
        status = "OK" if sr is not None else "READ_ERR"
        if status == "OK":
            read_ok += 1
            dur_total += float(dur)
            sr_hist[sr] = sr_hist.get(sr, 0) + 1
            if rdb is not None and rdb < args.silence_rms_db:
                near_sil += 1
                status = "NEAR_SILENCE"

        csv_rows.append([
            i, str(p), 1 if status != "MISSING" else 0,
            sr if sr is not None else "",
            nf if nf is not None else "",
            ch if ch is not None else "",
            f"{dur:.3f}" if dur is not None else "",
            f"{rdb:.1f}" if rdb is not None else "",
            status
        ])

    # write CSV
    out_csv = Path(args.out_csv) if args.out_csv else man_path.with_suffix(".audit.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["idx","path","exists","sr","num_frames","channels","dur_s","probe_rms_db","status"])
        wr.writerows(csv_rows)

    # summary
    print(f"\n[verify] {man_path}")
    print(f"  rows                : {len(rows)}")
    print(f"  existing files      : {exists_ok}")
    print(f"  readable (probe OK) : {read_ok}")
    print(f"  near-silence (<{args.silence_rms_db:.1f} dB) : {near_sil}")
    print(f"  duration (sum hours): {dur_total/3600.0:.2f} h")
    if sr_hist:
        top = sorted(sr_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        print("  sample_rate hist    :", ", ".join([f"{sr} Hz: {n}" for sr, n in top]))
    print(f"  wrote CSV           : {out_csv}\n")

if __name__ == "__main__":
    main()
