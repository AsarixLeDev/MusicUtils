#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan MUSAN/DEMAND (or any folder) and emit a JSONL noise manifest with lightweight filtering.
Filtering is keyword-based (path substrings) as defined in configs/noise_filters.yaml.

Each row: {"noise": "/path/to/file.wav", "sr_src": 48000, "dur_s": 12.3, "label": "room_tone", "source": "musan"}
"""

import argparse, json, sys, os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml, torchaudio

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac"}

def audio_info(path: Path) -> Optional[Dict[str, Any]]:
    try:
        info = torchaudio.info(str(path))
        sr = info.sample_rate
        dur_s = info.num_frames / float(sr) if sr > 0 else 0.0
        return {"sr": sr, "dur_s": dur_s}
    except Exception:
        return None

def allowed_by_filters(path: Path, filters: Dict[str, Any]) -> bool:
    s = str(path).lower().replace("\\", "/")
    inc = set([x.lower() for x in filters.get("include_labels", [])])
    exc = set([x.lower() for x in filters.get("exclude_labels", [])])
    if exc and any(x in s for x in exc):
        return False
    if inc and not any(x in s for x in inc):
        return False
    return True

def infer_label(path: Path) -> str:
    # crude: use parent folder name as label
    return path.parent.name.lower()

def scan_dir(root: Path, filters: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in AUDIO_EXTS: continue
        if not allowed_by_filters(p, filters): continue
        meta = audio_info(p)
        if not meta: continue

        # constraints from filters
        cons = filters.get("constraints", {})
        min_dur = float(cons.get("min_dur_s", 0.0))
        max_dur = float(cons.get("max_dur_s", 10**9))
        if not (min_dur <= meta["dur_s"] <= max_dur):
            continue

        rows.append({
            "noise": str(p),
            "sr_src": meta["sr"],
            "dur_s": meta["dur_s"],
            "label": infer_label(p),
            "source": source_name,
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/data_paths_example.yaml",
                    help="YAML with musan.root / demand.root and manifests_out.*")
    ap.add_argument("--filters", type=str, default="../configs/noise_filters.yaml")
    ap.add_argument("--dataset", type=str, choices=["musan", "demand", "any"], default="demand")
    ap.add_argument("--root", type=str, default="", help="if dataset=any, root to scan")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.filters, "r", encoding="utf-8") as f:
        filt = yaml.safe_load(f)

    if args.dataset == "musan":
        root = Path(cfg["musan"]["root"])
        src_name = "musan"
    elif args.dataset == "demand":
        root = Path(cfg["demand"]["root"])
        src_name = "demand"
    else:
        root = Path(args.root) if args.root else None
        src_name = "custom"
    if not root or not root.exists():
        print(f"[ERR] root not found for dataset={args.dataset}", file=sys.stderr); sys.exit(1)

    rows = scan_dir(root, filt, src_name)
    if not rows:
        print("[WARN] No rows collected.")
        return

    if args.out:
        out = Path(args.out)
    else:
        out_dir = Path(cfg["manifests_out"]["dir"])
        out = out_dir / ("musan_musicnoise.jsonl" if args.dataset == "musan" else "demand.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append and out.exists() else "w"
    with out.open(mode, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
