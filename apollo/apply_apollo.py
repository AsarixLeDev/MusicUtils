#!/usr/bin/env python3
# apply_apollo.py — thin wrapper that calls the base-level chunked runner
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input WAV")
    ap.add_argument("--out", dest="outp", required=True, help="output WAV")
    ap.add_argument("--repo_dir", default="./.apollo_repo", help="where to clone/use the Apollo code")
    ap.add_argument("--chunk_sec", type=float, default=2.5)
    ap.add_argument("--overlap_sec", type=float, default=0.4)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--fp16", action="store_true", help="Enable AMP on CUDA (off by default)")
    args = ap.parse_args()

    base = Path(__file__).parent.resolve()
    runner = base / "inference_chunked.py"
    if not runner.exists():
        raise SystemExit(f"Missing {runner}. Place inference_chunked.py in the same folder as apply_apollo.py")

    env = os.environ.copy()
    # Helps reduce fragmentation on some Windows + NVIDIA setups
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        sys.executable, str(runner),
        "--in_wav", str(Path(args.inp).resolve()),
        "--out_wav", str(Path(args.outp).resolve()),
        "--repo_dir", str(Path(args.repo_dir).resolve()),
        "--chunk_sec", str(args.chunk_sec),
        "--overlap_sec", str(args.overlap_sec),
        "--device", args.device,
    ]
    if args.mono:
        cmd.append("--mono")
    if args.fp16:
        cmd.append("--fp16")

    print("[apollo] Running inference …")
    subprocess.run(cmd, check=True, env=env)
    print("[apollo] Done.")

if __name__ == "__main__":
    main()
