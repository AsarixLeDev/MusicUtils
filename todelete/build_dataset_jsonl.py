#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, json, time, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".w64"}

# Lazy imports for speed (only when needed)
def try_import_soundfile():
    try:
        import soundfile as sf
        return sf
    except Exception:
        return None

def try_import_torchaudio():
    try:
        import torchaudio
        return torchaudio
    except Exception:
        return None

def try_import_librosa():
    try:
        import librosa
        return librosa
    except Exception:
        return None

def posix(p: Path) -> str:
    return str(p.resolve().as_posix())

def walk_audio(root: Path) -> List[Path]:
    if not root or not root.exists():
        return []
    files: List[Path] = []
    stack = [root]
    exts = AUDIO_EXT
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    elif e.is_file():
                        sfx = os.path.splitext(e.name)[1].lower()
                        if sfx in exts:
                            files.append(Path(e.path))
        except (PermissionError, FileNotFoundError):
            continue
    return files

# ---------- duration probing with cache ----------

def load_cache(path: Path) -> Dict[str, Dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(path: Path, cache: Dict[str, Dict]):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass

def cache_key(p: Path) -> str:
    try:
        st = p.stat()
        return f"{posix(p)}::{st.st_size}::{int(st.st_mtime)}"
    except Exception:
        return f"{posix(p)}::0::0"

def probe_duration(path: Path, mode: str = "fast") -> float:
    """
    mode: 'none' (return 0), 'fast' (soundfile+torchaudio), 'full' (+librosa fallback)
    """
    if mode == "none":
        return 0.0

    # soundfile.info (fast, no full decode)
    sf = try_import_soundfile()
    if sf is not None:
        try:
            info = sf.info(str(path))
            if info.frames and info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass

    # torchaudio.info (also cheap)
    ta = try_import_torchaudio()
    if ta is not None:
        try:
            i = ta.info(str(path))
            if getattr(i, "num_frames", 0) and getattr(i, "sample_rate", 0):
                return float(i.num_frames) / float(i.sample_rate)
        except Exception:
            pass

    if mode != "full":
        return 0.0

    # librosa fallback (can be slow; only if asked)
    lb = try_import_librosa()
    if lb is not None:
        try:
            d = lb.get_duration(filename=str(path))
            return float(d or 0.0)
        except Exception:
            pass

    return 0.0

def parallel_probe(paths: List[Path], mode: str, workers: int, cache: Dict[str, Dict], log_prefix: str = "") -> Dict[str, float]:
    """
    Returns map path->duration (seconds). Uses/updates cache {key: {dur, path}}.
    """
    t0 = time.time()
    out: Dict[str, float] = {}
    todo: List[Tuple[str, Path]] = []
    for p in paths:
        k = cache_key(p)
        c = cache.get(k)
        if c is not None:
            out[posix(p)] = float(c.get("dur", 0.0))
        else:
            todo.append((k, p))

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            fut = {ex.submit(probe_duration, p, mode): (k, p) for (k, p) in todo}
            for f in as_completed(fut):
                k, p = fut[f]
                dur = 0.0
                try:
                    dur = max(0.0, float(f.result()))
                except Exception:
                    dur = 0.0
                out[posix(p)] = dur
                cache[k] = {"dur": dur, "path": posix(p)}

    if log_prefix:
        took = time.time() - t0
        print(f"[probe:{log_prefix}] {len(paths)} files, mode={mode}, workers={workers}, time={took:.1f}s (cache hits: {len(paths)-len(todo)})")
    return out

# ---------- dataset-specific scanners ----------

def scan_musdb18hq(root: Path, use_vocals: bool, use_mixtures: bool) -> List[Path]:
    out: List[Path] = []
    for split in ("train", "test"):
        sdir = root / split
        if not sdir.exists():
            continue
        for track in sdir.iterdir():
            if not track.is_dir():
                continue
            if use_vocals:
                v = track / "vocals.wav"
                if v.exists(): out.append(v)
            if use_mixtures:
                m = track / "mixture.wav"
                if m.exists(): out.append(m)
    return out

def scan_moisesdb(root: Path, use_vocals: bool, use_mixtures: bool) -> List[Path]:
    out: List[Path] = []
    if not root.exists(): return out
    for track in root.iterdir():
        if not track.is_dir(): continue
        if use_vocals:
            vdir = track / "vocals"
            if vdir.exists():
                out.extend(walk_audio(vdir))
        if use_mixtures:
            mdir = track / "mixture"
            if mdir.exists():
                out.extend(walk_audio(mdir))
    return out

def scan_musan_noise(root: Path, include_music=False, include_speech=False) -> List[Path]:
    out: List[Path] = []
    if not root.exists(): return out
    out += walk_audio(root / "noise")
    if include_music: out += walk_audio(root / "music")
    if include_speech: out += walk_audio(root / "speech")
    return out

def scan_demand_noise(root: Path) -> List[Path]:
    return walk_audio(root)

# ---------- manifest builders ----------

def build_clean_rows(clean_paths: List[Path], dur_map: Dict[str, float], min_dur: float) -> List[Dict]:
    rows: List[Dict] = []
    for p in clean_paths:
        q = posix(p)
        d = float(dur_map.get(q, 0.0))
        if d >= min_dur:
            rows.append({"clean": q, "duration": round(d, 3), "src": "clean"})
    return rows

def attach_noise_pool(rows: List[Dict], noise_paths: List[Path], max_noises_per_item: int, seed: int) -> List[Dict]:
    if not noise_paths:
        return rows
    import random
    rng = random.Random(seed)
    all_noise = sorted(set(posix(p) for p in noise_paths))
    if max_noises_per_item and max_noises_per_item > 0:
        out = []
        for r in rows:
            pool = all_noise.copy()
            rng.shuffle(pool)
            r2 = dict(r)
            r2["noises"] = pool[:max_noises_per_item]
            out.append(r2)
        return out
    else:
        for r in rows:
            r["noises"] = all_noise
        return rows

def split_train_val(rows: List[Dict], val_ratio: float, val_count: Optional[int], seed: int) -> Tuple[List[Dict], List[Dict]]:
    import random
    # de-dup rows by JSON key
    rows = list({json.dumps(r, sort_keys=True): r for r in rows}.values())
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = min(len(rows), val_count) if (val_count and val_count > 0) else int(round(val_ratio * len(rows)))
    return rows[n_val:], rows[:n_val]

def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {path}  ({len(rows)} lines)")

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Fast builder for UTF-8 JSONL manifests (MoisesDB/MUSDB18HQ + MUSAN/DEMAND noise)")
    # roots
    ap.add_argument("--musdb", type=str, default="")
    ap.add_argument("--moisesdb", type=str, default="")
    ap.add_argument("--musan", type=str, default="")
    ap.add_argument("--demand", type=str, default="")
    ap.add_argument("--extra-noise-dirs", nargs="*", default=[])
    # what to include as CLEAN
    ap.add_argument("--use-vocals", action="store_true")
    ap.add_argument("--use-mixtures", action="store_true")
    # probing & speed
    ap.add_argument("--probe", choices=["none","fast","full"], default="fast",
                    help="Control duration probing. 'none' is fastest; 'fast' uses soundfile+torchaudio; 'full' adds librosa fallback.")
    ap.add_argument("--probe-noise", action="store_true", help="Also probe noise files (usually unnecessary).")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8)//2))
    ap.add_argument("--cache", type=str, default="", help="Duration cache path (default near --out-train)")
    # filtering / pool sizing
    ap.add_argument("--min-dur", type=float, default=4.0)
    ap.add_argument("--max-noises-per-item", type=int, default=12)
    # split/output
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--val-count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-train", type=str, required=True)
    ap.add_argument("--out-val", type=str, required=True)
    # musan options
    ap.add_argument("--include-musan-music", action="store_true")
    ap.add_argument("--include-musan-speech", action="store_true")
    return ap.parse_args()

def main():
    a = parse_args()
    # resolve paths
    musdb   = Path(os.path.expanduser(os.path.expandvars(a.musdb)))   if a.musdb   else None
    moises  = Path(os.path.expanduser(os.path.expandvars(a.moisesdb))) if a.moisesdb else None
    musan   = Path(os.path.expanduser(os.path.expandvars(a.musan)))    if a.musan   else None
    demand  = Path(os.path.expanduser(os.path.expandvars(a.demand)))   if a.demand  else None
    extraN  = [Path(os.path.expanduser(os.path.expandvars(p))) for p in a.extra_noise_dirs]

    # collect CLEAN paths quickly
    clean_paths: List[Path] = []
    if musdb:
        clean_paths += scan_musdb18hq(musdb, use_vocals=a.use_vocals, use_mixtures=a.use_mixtures)
    if moises:
        clean_paths += scan_moisesdb(moises, use_vocals=a.use_vocals, use_mixtures=a.use_mixtures)
    clean_paths = sorted(set(clean_paths))
    print(f"[clean] candidates: {len(clean_paths)}")

    # collect NOISE paths quickly
    noise_paths: List[Path] = []
    if musan:
        noise_paths += scan_musan_noise(musan, include_music=a.include_musan_music, include_speech=a.include_musan_speech)
    if demand:
        noise_paths += scan_demand_noise(demand)
    for nd in extraN:
        noise_paths += walk_audio(nd)
    noise_paths = sorted(set(noise_paths))
    print(f"[noise] candidates: {len(noise_paths)}")

    # load duration cache (fast)
    cache_path = Path(a.cache) if a.cache else Path(a.out_train).with_suffix(".dur_cache.json")
    cache = load_cache(cache_path)

    # probe durations (CLEAN always, NOISE optional)
    dur_clean = parallel_probe(clean_paths, mode=a.probe, workers=a.workers, cache=cache, log_prefix="clean")
    dur_noise = {}
    if a.probe_noise and noise_paths:
        dur_noise = parallel_probe(noise_paths, mode=a.probe, workers=a.workers, cache=cache, log_prefix="noise")

    # persist cache
    save_cache(cache_path, cache)

    # build rows
    rows = build_clean_rows(clean_paths, dur_clean, min_dur=max(0.0, a.min_dur))
    rows = attach_noise_pool(rows, noise_paths, max_noises_per_item=a.max_noises_per_item, seed=a.seed)

    # split
    train, val = split_train_val(rows, val_ratio=a.val_ratio, val_count=(a.val_count if a.val_count>0 else None), seed=a.seed)

    # write
    write_jsonl(Path(a.out_train), train)
    write_jsonl(Path(a.out_val),   val)

if __name__ == "__main__":
    main()
