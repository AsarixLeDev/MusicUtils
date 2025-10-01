#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan MUSDB‑HQ directory, emit JSONL manifest with mixture as 'clean' and optional stems.
Works with your current loader (we only consume 'clean' now).
"""

import argparse, json, sys, os
from pathlib import Path
from typing import Dict, Any, Optional
import torchaudio
import yaml

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac"}

def audio_info(path: Path) -> Optional[Dict[str, Any]]:
    try:
        info = torchaudio.info(str(path))
        sr = info.sample_rate
        dur_s = info.num_frames / float(sr) if sr > 0 else 0.0
        return {"sr": sr, "dur_s": dur_s}
    except Exception:
        return None

def find_track_items(track_dir: Path) -> Optional[Dict[str, Path]]:
    # MUSDB-HQ canonical names
    mix = track_dir / "mixture.wav"
    stems = {
        "vocals": track_dir / "vocals.wav",
        "drums":  track_dir / "drums.wav",
        "bass":   track_dir / "bass.wav",
        "other":  track_dir / "other.wav",
    }
    if not mix.exists():
        return None
    for k, p in stems.items():
        if not p.exists():
            # tolerate missing stems; we only need mixture now
            pass
    return {"mix": mix, "stems": stems}

def write_jsonl(out_path: Path, rows, append: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./configs/data_paths_example.yaml",
                    help="YAML with musdb_hq.root and manifests_out.*")
    ap.add_argument("--split", type=str, choices=["train", "val", "valid", "test"], default="train")
    ap.add_argument("--out",   type=str, default="", help="Override output JSONL path")
    ap.add_argument("--append", action="store_true", help="Append to existing JSONL")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["musdb_hq"]["root"])
    split_map = {"valid": "valid", "val": "valid", "test": "test", "train": "train"}
    sub = split_map.get(args.split, "train")
    split_dir = root / sub
    if not split_dir.exists():
        print(f"[ERR] MUSDB-HQ split not found: {split_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for track in sorted(split_dir.iterdir()):
        if not track.is_dir(): continue
        items = find_track_items(track)
        if not items: continue
        info = audio_info(items["mix"])
        if not info:
            print(f"[WARN] Could not read {items['mix']}", file=sys.stderr)
            continue

        # manifest row with mixture as 'clean'
        row = {
            "clean": str(items["mix"]),
            "sr_src": info["sr"],
            "dur_s": info["dur_s"],
            "source": "musdbhq",
            "split": sub,
            "track": track.name,
            "stems": {k: str(p) for k, p in items["stems"].items() if p.exists()},
        }
        rows.append(row)

    if not rows:
        print("[WARN] No rows collected; nothing written.")
        return

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(cfg["manifests_out"]["dir"])
        fname = cfg["manifests_out"]["music_stems_train"] if sub == "train" else cfg["manifests_out"]["music_stems_val"]
        out_path = out_dir / fname

    write_jsonl(out_path, rows, append=args.append)
    print(f"[OK] Wrote {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()
