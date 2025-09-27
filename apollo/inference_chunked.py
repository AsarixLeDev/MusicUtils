#!/usr/bin/env python3
# inference_chunked.py — base-level runner (no edits inside .apollo_repo needed)
import argparse
import sys
import subprocess
from pathlib import Path

import torch

# Prefer torchaudio; fall back to soundfile
try:
    import torchaudio
except Exception:
    torchaudio = None
    import soundfile as sf
    import numpy as np

def ensure_pkg(mod_name, pip_name=None, py=sys.executable):
    try:
        __import__(mod_name)
    except Exception:
        subprocess.run([py, "-m", "pip", "install", "--upgrade", pip_name or mod_name], check=True)

def ensure_repo(repo_dir: Path):
    repo_url = "https://github.com/JusperLee/Apollo.git"
    if not repo_dir.exists():
        print(f"[apollo] Cloning {repo_url} -> {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    # Make it importable
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

def load_audio(path: Path, target_sr=44100):
    if torchaudio is not None:
        wav, sr = torchaudio.load(str(path))        # [C, T], float32
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        return wav, sr
    else:
        wav, sr = sf.read(str(path), always_2d=True)  # [T, C]
        wav = wav.astype("float32")
        if sr != target_sr:
            raise RuntimeError("Resampling needs torchaudio; provide 44.1kHz input or install torchaudio.")
        return torch.from_numpy(wav.T), sr

def save_audio(path: Path, wav: torch.Tensor, sr: int):
    wav = torch.clamp(wav.detach().cpu(), -1.0, 1.0)  # [C, T]
    if torchaudio is not None:
        torchaudio.save(str(path), wav, sr)
    else:
        sf.write(str(path), wav.numpy().T, sr)

def load_model(repo_dir: Path, device: torch.device):
    # deps
    ensure_pkg("huggingface_hub")
    from huggingface_hub import hf_hub_download

    ensure_repo(repo_dir)
    # now the repo is importable
    import look2hear.models as L2H

    ckpt_path = hf_hub_download("JusperLee/Apollo", filename="pytorch_model.bin")
    model = L2H.BaseModel.from_pretrain(
        ckpt_path, sr=44100, win=20, feature_dim=256, layer=6
    ).to(device)
    model.eval()
    return model

@torch.no_grad()
def infer_overlap_add(
    model: torch.nn.Module,
    wav: torch.Tensor,  # [C, T] on CPU
    sr: int,
    device: torch.device,
    chunk_sec: float = 2.5,
    overlap_sec: float = 0.4,
    mono: bool = False,
    use_fp16: bool = False,   # default False to avoid cuFFT half-precision issue (n_fft=882)
):
    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    C, T = wav.shape
    L_in = max(1, int(round(chunk_sec * sr)))
    O = max(0, int(round(overlap_sec * sr)))

    # Accumulators with a little tail room
    acc = torch.zeros(1, C, T + L_in, device=device)
    wsum = torch.zeros(1, 1, T + L_in, device=device)

    i = 0
    while i < T:
        chunk = wav[:, i: i + L_in]  # [C, l_in]
        l_in = chunk.shape[-1]

        # ---- SAFE tail pad: reflect if possible, else zero-pad ----
        if l_in < L_in:
            pad_amt = L_in - l_in
            if l_in > 1 and pad_amt <= l_in - 1:
                chunk = torch.nn.functional.pad(chunk, (0, pad_amt), mode="reflect")
            else:
                # too short to reflect-pad; zero-pad the tail
                chunk = torch.nn.functional.pad(chunk, (0, pad_amt), mode="constant", value=0.0)

        x = chunk.unsqueeze(0).to(device)  # [1, C, L_in]

        # ---- forward (AMP optional; auto-fallback to fp32 for cuFFT limits) ----
        try:
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=(device.type == "cuda" and use_fp16),
            ):
                y = model(x)  # [1, C, l_out] or [1, l_out]
        except RuntimeError as e:
            if "cuFFT" in str(e) and "half precision" in str(e):
                with torch.autocast(device_type=device.type, enabled=False):
                    y = model(x)
            else:
                raise

        # Normalize to [1, C, l_out]
        if y.dim() == 2:
            y = y.unsqueeze(1)  # [1, 1, T]
        elif y.dim() != 3:
            raise RuntimeError(f"Unexpected output shape: {tuple(y.shape)}")

        l_out = y.shape[-1]

        # Hann window matching the **output** length
        w = torch.hann_window(l_out, periodic=False, device=device).view(1, 1, -1)

        # Cap overlap to <= half of this chunk's output for safety
        o = min(O, l_out // 2)
        acc[..., i: i + l_out] += y * w
        wsum[..., i: i + l_out] += w

        step_eff = max(1, l_out - o)
        i += step_eff

    out = acc[..., :T] / torch.clamp_min(wsum[..., :T], 1e-8)  # [1, C, T]
    return out.squeeze(0)  # [C, T]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--repo_dir", default="./.apollo_repo")
    ap.add_argument("--chunk_sec", type=float, default=6.0)
    ap.add_argument("--overlap_sec", type=float, default=0.5)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--fp16", action="store_true", help="Enable AMP on CUDA (off by default)")
    args = ap.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device("cuda") if args.device == "cuda" and torch.cuda.is_available()
        else torch.device("cpu")
    )

    repo_dir = Path(args.repo_dir).resolve()
    model = load_model(repo_dir, device)

    wav, sr = load_audio(Path(args.in_wav), target_sr=44100)

    try:
        out = infer_overlap_add(
            model, wav, sr, device,
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
            mono=args.mono,
            use_fp16=args.fp16,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[apollo] OOM on GPU — retrying on CPU.")
        device = torch.device("cpu")
        model = load_model(repo_dir, device)
        out = infer_overlap_add(
            model, wav, sr, device,
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
            mono=args.mono,
            use_fp16=False,
        )

    save_audio(Path(args.out_wav), out, sr)
    print(f"[apollo] Saved: {args.out_wav} @ {sr} Hz")

if __name__ == "__main__":
    main()
