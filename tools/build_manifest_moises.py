#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a Moises-style dataset and emit a JSONL manifest with the mixture as 'clean'.

This script is robust to:
- unknown stem names (labels are inferred from the first directory under each track),
- multiple stems per label (e.g., guitar/*.wav chunks),
- 'mixture' being inside a 'mixture/' subfolder OR named like 'mix|mixture|master|accompaniment' elsewhere.

Output (one line per track):
{
  "clean": "<abs path to mixture>",
  "sr_src": 48000,
  "dur_s": 12.34,
  "source": "moises",
  "split": "train"|"val",
  "track": "<relative track id>",
  "stems": [{"label":"guitar","path":".../guitar/abc.wav"}, ...],
  "num_stems": 17,
  "stems_by_label": {"guitar": 8, "bass": 1, ...}
}

Usage:
  python tools/build_manifest_moises.py \
    --config configs/data_paths_example.yaml \
    --split train \
    --out manifests/music_stems_train.jsonl
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio
import yaml


# -------- config / defaults --------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
MIX_PRI_KEYWORDS = ("mixture", "mix", "master", "accompaniment")


def audio_info(p: Path) -> Optional[Dict[str, float]]:
    """Return {'sr': int, 'dur_s': float} or None on failure."""
    try:
        i = torchaudio.info(str(p))
        sr = int(i.sample_rate)
        dur_s = float(i.num_frames) / max(1, sr)
        return {"sr": sr, "dur_s": dur_s}
    except Exception:
        return None


def rank_mixture_candidate(files: List[Path]) -> Optional[Path]:
    """
    Choose the best mixture candidate from a list of audio files in a track dir.
    Preference:
      1) path segments include a folder named 'mixture' AND file named 'mixture.*'
      2) filename contains one of MIX_PRI_KEYWORDS
      3) fallback: shortest filename (heuristic)
    """
    if not files:
        return None

    # 1) mixture/mixture.*
    for f in files:
        parts = [p.lower() for p in f.parts]
        if "mixture" in parts and f.name.lower().startswith("mixture."):
            return f

    # 2) filename contains a preferred keyword (highest count wins; then shorter name)
    ranked = []
    for f in files:
        name = f.name.lower()
        score = sum(1 for k in MIX_PRI_KEYWORDS if k in name)
        ranked.append((score, len(name), f))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if ranked and ranked[0][0] > 0:
        return ranked[0][2]

    # 3) fallback: shortest filename
    files_sorted = sorted(files, key=lambda p: len(p.name))
    return files_sorted[0]


def collect_track_dirs(root: Path) -> List[Path]:
    """
    Build a list of candidate track directories by:
      - finding any 'mixture' subfolder that contains audio,
      - OR directories that contain at least one audio file.
    We return the **highest** directory that likely represents a single track.
    """
    # Fast path: all dirs that contain 'mixture' folder with audio
    tracks = set()
    for mix_dir in root.rglob("mixture"):
        if not mix_dir.is_dir():
            continue
        wavs = [p for p in mix_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS]
        if wavs:
            tracks.add(mix_dir.parent)

    # Fallback: any dir with audio files gets accepted as a track
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if any((p.suffix.lower() in AUDIO_EXTS) for p in d.iterdir()):
            # try to avoid duplicating subdirs under an already added track
            # keep only the highest parent: if a parent is in tracks, skip
            parent_included = False
            for t in tracks:
                try:
                    d.relative_to(t)
                    parent_included = True
                    break
                except Exception:
                    pass
            if not parent_included:
                tracks.add(d)

    # Return deterministic order
    return sorted(tracks)


def gather_audio_files(track_dir: Path) -> Tuple[List[Path], Dict[str, List[Path]]]:
    """
    Return (all_audio_files, stems_by_label)
    - The 'label' is the **first subfolder** under the track dir (e.g. guitar/, drums/, vocals/, other/…).
    - All audio files under 'mixture/' are **excluded** from stems.
    """
    files = []
    stems_by_label: Dict[str, List[Path]] = defaultdict(list)

    for p in track_dir.rglob("*"):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        files.append(p)

    # Build stems_by_label from first-level folder under the track
    for p in files:
        # skip mixture files from stems
        if "mixture" in [seg.lower() for seg in p.parts]:
            continue
        # label from first segment after track_dir
        try:
            rel = p.relative_to(track_dir)
        except Exception:
            # Shouldn't happen; be conservative
            rel = p
        label = rel.parts[0].lower() if len(rel.parts) > 1 else "_root"
        stems_by_label[label].append(p)

    return files, stems_by_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/data_paths_example.yaml",
                    help="YAML with moises.root and manifests_out.*")
    ap.add_argument("--split", type=str, choices=["train", "test"], default="train")
    ap.add_argument("--out", type=str, default="", help="Override output JSONL path")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--min-dur-s", type=float, default=0.2,
                    help="Skip very short mixtures (< this many seconds)")
    args = ap.parse_args()

    # Load paths
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg["moises"]["root"])
        if not root.exists():
            print(f"[ERR] Moises root not found: {root}", file=sys.stderr)
            sys.exit(1)
        out_path = Path(args.out) if args.out else Path(cfg["manifests_out"]["dir"]) / \
                   ("music_stems_train.jsonl" if args.split == "train" else "music_stems_test.jsonl")
    except Exception as e:
        print(f"[ERR] bad config or paths: {e}", file=sys.stderr)
        sys.exit(2)

    # Find track dirs
    track_dirs = collect_track_dirs(root)
    if not track_dirs:
        print("[WARN] no track directories found")
        return

    n_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as out_f:
        for td in track_dirs:
            files, stems_by_label = gather_audio_files(td)
            if not files:
                continue

            # Locate mixture
            mixture = rank_mixture_candidate(files)
            if mixture is None:
                # No mixture found; skip this track
                continue

            # Mixture info & duration filter
            info = audio_info(mixture)
            if not info:
                continue
            if info["dur_s"] < float(args.min_dur_s):
                continue

            # Build stems list (label, path)
            stems_list = []
            by_label_count = {}
            for label, plist in stems_by_label.items():
                # Exclude mixture folder stem label if any leaked
                if label.lower() == "mixture":
                    continue
                # Sort stems for determinism
                plist_sorted = sorted(plist)
                by_label_count[label] = len(plist_sorted)
                for p in plist_sorted:
                    stems_list.append({"label": label, "path": str(p)})

            row = {
                "clean": str(mixture),
                "sr_src": info["sr"],
                "dur_s": info["dur_s"],
                "source": "moises",
                "split": args.split,
                "track": str(td.relative_to(root)).replace("\\", "/"),
                "stems": stems_list,                    # list of {label, path}
                "num_stems": len(stems_list),
                "stems_by_label": by_label_count,       # label -> count
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] Wrote {n_written} rows -> {out_path}")


if __name__ == "__main__":
    main()
