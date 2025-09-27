#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

from tqdm import tqdm

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


# -------------------- encoding helpers --------------------

def detect_text_encoding(p: Path) -> str:
    """
    Very small heuristic: BOM check, then try utf-8, utf-16, utf-16-le, utf-16-be, utf-8-sig, latin-1.
    Returns a Python codec name.
    """
    with p.open("rb") as f:
        head = f.read(4)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if head.startswith(b"\xfe\xff"):
        return "utf-16-be"
    # try candidates
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            _ = p.read_text(encoding=enc)
            return enc
        except Exception:
            continue
    # last resort
    return "latin-1"


def convert_manifest_to_utf8(src: Path, dst: Path) -> Tuple[int, int]:
    """
    Reads src with detected encoding and writes dst as UTF-8, one JSON per line.
    Returns (num_lines, num_ok_json).
    """
    enc = detect_text_encoding(src)
    text = src.read_text(encoding=enc, errors="strict")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    ok = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as fw:
        for i, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
                fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                print(f"[warn] {src.name}:{i} not valid JSON; skipping: {e}")
                continue
    return len(lines), ok


# -------------------- scanning helpers --------------------

def list_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT]


def map_by_stem(files: Iterable[Path]) -> Dict[str, Path]:
    d: Dict[str, Path] = {}
    for p in files:
        st = p.stem
        # If duplicate stems exist, keep the shortest path (heuristic)
        if st not in d or len(str(p)) < len(str(d[st])):
            d[st] = p
    return d


def posix(p: Path) -> str:
    return str(p.resolve().as_posix())


# -------------------- builders --------------------

def build_pairs(clean_dir: Path, noisy_dir: Path) -> List[Dict]:
    clean_map = map_by_stem(list_audio_files(clean_dir))
    noisy_map = map_by_stem(list_audio_files(noisy_dir))
    stems = sorted(set(clean_map) & set(noisy_map))
    rows: List[Dict] = []
    for st in tqdm(stems, desc="pairing", dynamic_ncols=True):
        rows.append({"clean": posix(clean_map[st]), "noisy": posix(noisy_map[st])})
    return rows


def build_clean_with_noises(clean_dirs: List[Path], noise_dirs: List[Path], max_noises_per_item: int | None = None) -> \
List[Dict]:
    clean_files: List[Path] = []
    for d in clean_dirs:
        clean_files.extend(list_audio_files(d))
    noise_files: List[Path] = []
    for d in noise_dirs:
        noise_files.extend(list_audio_files(d))

    noise_paths = [posix(p) for p in noise_files]
    noise_paths = sorted(list(set(noise_paths)))  # de-dupe

    rows: List[Dict] = []
    for p in tqdm(clean_files, desc="clean+noises", dynamic_ncols=True):
        rec: Dict = {"clean": posix(p)}
        if noise_paths:
            if max_noises_per_item is not None and max_noises_per_item > 0:
                # sample a fixed subset per item but keep deterministic by hash
                import random
                rnd = random.Random(hash(rec["clean"]) & 0xFFFFFFFF)
                pool = noise_paths.copy()
                rnd.shuffle(pool)
                rec["noises"] = pool[:max_noises_per_item]
            else:
                rec["noises"] = noise_paths
        rows.append(rec)
    return rows


# -------------------- train/val split --------------------

def split_train_val(rows: List[Dict], val_ratio: float, val_count: int | None, seed: int) -> Tuple[
    List[Dict], List[Dict]]:
    import random
    rows = list(rows)
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if val_count is not None and val_count > 0:
        n_val = min(val_count, len(rows))
    else:
        n_val = int(round(val_ratio * len(rows)))
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val


# -------------------- write JSONL --------------------

def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------- validation --------------------

def quick_validate_audio(rows: List[Dict], max_check: int = 200) -> None:
    """
    Optional lightweight check to catch broken paths early.
    Does not read full audio (avoid slow), only checks file exists.
    """
    import itertools
    cnt = 0
    bad = 0
    for r in itertools.islice(rows, 0, max_check):
        for key in ("noisy", "clean", "noise"):
            if key in r:
                if not Path(r[key]).exists():
                    print(f"[warn] missing path: {r[key]}")
                    bad += 1
        if "noises" in r:
            for z in r["noises"][:5]:
                if not Path(z).exists():
                    print(f"[warn] missing noise path: {z}")
                    bad += 1
        cnt += 1
    if bad == 0:
        print(f"[validate] first {cnt} items look OK (paths exist).")


# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Build or convert UTF-8 JSONL manifests for denoiser training")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert existing manifest encoding to utf-8
    c = sub.add_parser("convert", help="Convert an existing JSONL (e.g., UTF-16) to UTF-8 JSONL")
    c.add_argument("--in", dest="inp", required=True, help="input manifest")
    c.add_argument("--out", dest="outp", required=True, help="output manifest (UTF-8)")

    # scan dataset folders and create manifests
    s = sub.add_parser("scan", help="Scan folders and create train/val JSONL (UTF-8)")
    s.add_argument("--pairs", nargs=2, metavar=("CLEAN_DIR", "NOISY_DIR"),
                   help="two directories: clean and noisy (matched by filename stem)")
    s.add_argument("--clean-dirs", nargs="+", default=[], help="one or more clean directories")
    s.add_argument("--noise-dirs", nargs="*", default=[], help="zero or more noise directories to attach as pool")
    s.add_argument("--max-noises-per-item", type=int, default=0, help="0=attach all noises; >0=limit per item")
    s.add_argument("--val-ratio", type=float, default=0.075,
                   help="validation split ratio (ignored if --val-count given)")
    s.add_argument("--val-count", type=int, default=0, help="fixed number of validation items")
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--out-train", required=True, help="output train.jsonl")
    s.add_argument("--out-val", required=True, help="output val.jsonl")
    s.add_argument("--validate", action="store_true", help="quick path existence checks on a sample")

    return ap.parse_args()


def main():
    a = parse_args()

    if a.cmd == "convert":
        src = Path(a.inp);
        dst = Path(a.outp)
        n, ok = convert_manifest_to_utf8(src, dst)
        print(f"[convert] {src} ({detect_text_encoding(src)}) -> {dst} (utf-8). Lines: {n}, valid JSON: {ok}")
        return

    if a.cmd == "scan":
        rows: List[Dict] = []
        if a.pairs:
            cdir = Path(a.pairs[0]);
            ndir = Path(a.pairs[1])
            if not cdir.exists() or not ndir.exists():
                raise FileNotFoundError("CLEAN_DIR or NOISY_DIR not found")
            print(f"[scan] pairing by stem:\n  clean={cdir}\n  noisy={ndir}")
            rows.extend(build_pairs(cdir, ndir))

        if a.clean_dirs:
            cdirs = [Path(p) for p in a.clean_dirs]
            if not all(d.exists() for d in cdirs):
                missing = [str(d) for d in cdirs if not d.exists()]
                raise FileNotFoundError(f"missing clean_dirs: {missing}")
            ndirs = [Path(p) for p in a.noise_dirs]
            print(f"[scan] clean+noises:\n  clean_dirs={cdirs}\n  noise_dirs={ndirs}")
            rows.extend(
                build_clean_with_noises(cdirs, ndirs, a.max_noises_per_item if a.max_noises_per_item > 0 else None))

        if not rows:
            print("No rows produced. Provide --pairs or --clean-dirs (with optional --noise-dirs).")
            sys.exit(2)

        # de-dupe identical dicts (by JSON string)
        uniq: Dict[str, Dict] = {}
        for r in rows:
            key = json.dumps(r, sort_keys=True)
            uniq[key] = r
        rows = list(uniq.values())

        train, val = split_train_val(rows, val_ratio=a.val_ratio, val_count=(a.val_count if a.val_count > 0 else None),
                                     seed=a.seed)
        out_tr = Path(a.out_train);
        out_va = Path(a.out_val)
        write_jsonl(out_tr, train);
        write_jsonl(out_va, val)
        print(f"[write] train={out_tr} ({len(train)} lines) | val={out_va} ({len(val)} lines)")

        if a.validate:
            quick_validate_audio(train + val)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user.")
    except Exception as e:
        print("[error]", e)
        traceback.print_exc()
        sys.exit(1)
