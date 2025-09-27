#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from soundrestorer.models.denoiser_net import ComplexUNet
from todelete.train.utils_stft import stft_pair, istft_from, hann_window

torch.backends.cudnn.benchmark = True


# ---------- Tone guard & Wiener ----------
def spectral_flatness(mag: torch.Tensor, eps=1e-12):
    gmean = torch.exp(torch.mean(torch.log(mag + eps), dim=1, keepdim=True))
    amean = torch.mean(mag + eps, dim=1, keepdim=True)
    return torch.clamp(gmean / amean, 0.0, 1.0)


def apply_tone_guard(M: torch.Tensor, X: torch.Tensor, guard=0.5, thresh=0.4):
    if guard <= 0: return M
    mag = X.abs().float() + 1e-8
    sf_ = spectral_flatness(mag)  # (B,1,T)
    tonal = (sf_ < thresh).float().expand(-1, mag.shape[1], -1)  # (B,F,T)
    alpha = guard * tonal
    Mr = (1 - alpha) * M[:, 0] + alpha * 1.0
    Mi = (1 - alpha) * M[:, 1] + alpha * 0.0
    return torch.stack([Mr, Mi], dim=1)


def wiener_refine(noisy: torch.Tensor, yhat: torch.Tensor, win: torch.Tensor, n_fft=1024, hop=256, kappa=1.0):
    if kappa <= 0: return yhat
    X, _ = stft_pair(noisy, win, n_fft=n_fft, hop=hop)
    Y, _ = stft_pair(yhat, win, n_fft=n_fft, hop=hop)
    R = X - Y
    Sxx = (Y.abs() ** 2).float()
    Nxx = R.abs().float().pow(2).median(dim=-1, keepdim=True).values
    G = Sxx / (Sxx + kappa * (Nxx + 1e-8))
    Z = torch.complex(G, torch.zeros_like(G)) * X
    return istft_from(Z, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)


# ---------- Resampling ----------
def resample_if_needed(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out: return x
    try:
        import torchaudio
        wav = torch.from_numpy(x.T).unsqueeze(0)  # (1,C,T)
        y = torchaudio.functional.resample(wav, sr_in, sr_out)
        return y.squeeze(0).numpy().T
    except Exception:
        try:
            from scipy.signal import resample_poly
            g = np.gcd(sr_in, sr_out)
            up, down = sr_out // g, sr_in // g
            return resample_poly(x, up, down, axis=0)
        except Exception:
            # linear fallback
            t = np.linspace(0, 1, num=x.shape[0], endpoint=False)
            t2 = np.linspace(0, 1, num=int(round(x.shape[0] * sr_out / sr_in)), endpoint=False)
            y = np.stack([np.interp(t2, t, x[:, c]) for c in range(x.shape[1])], axis=1)
            return y


# ---------- Chunked OLA ----------
def cosine_fade(n: int, device):
    t = torch.linspace(0, np.pi, steps=n, device=device)
    return (1 - torch.cos(t)) * 0.5


def process_waveform(
        net: ComplexUNet,
        audio: torch.Tensor,  # (C,T)
        sr: int,
        device: str,
        n_fft: int = 1024,
        hop: int = 256,
        mask_limit: float = 1.05,
        strength: float = 0.9,
        wiener_kappa: float = 1.5,
        tone_guard: float = 0.6,
        tone_thresh: float = 0.45,
        amp: str = "bfloat16",
        channels_last: bool = True,
        chunk_sec: float = 20.0,
        overlap_sec: float = 0.5,
) -> torch.Tensor:
    C, T = audio.shape
    win = hann_window(n_fft, device=device, dtype=torch.float32)
    amp_enabled = device.startswith("cuda") and amp != "float32"
    amp_dtype = torch.bfloat16 if amp == "bfloat16" else torch.float16
    hop_len = int(round(chunk_sec * sr - overlap_sec * sr))
    chunk_len = hop_len + int(round(overlap_sec * sr))
    fade_len = int(round(overlap_sec * sr / 2))
    fade_in = cosine_fade(fade_len, device)
    fade_out = torch.flip(fade_in, dims=[0])

    pos = 0
    chunks = []
    while pos < T:
        tail = min(chunk_len, T - pos)
        seg = audio[:, pos:pos + tail].to(device)
        if seg.shape[1] < chunk_len:
            seg = torch.cat([seg, torch.zeros(C, chunk_len - seg.shape[1], device=device)], dim=1)

        with torch.no_grad():
            with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"), dtype=amp_dtype,
                                enabled=amp_enabled):
                X, Xri = stft_pair(seg, win, n_fft=n_fft, hop=hop)
                if channels_last and device.startswith("cuda"):
                    Xri = Xri.contiguous(memory_format=torch.channels_last)
                M = net(Xri)
                if mask_limit and mask_limit > 0:
                    Mr_, Mi_ = M[:, 0], M[:, 1]
                    mag_ = torch.sqrt(Mr_ ** 2 + Mi_ ** 2 + 1e-8)
                    scale_ = torch.clamp(mask_limit / mag_, max=1.0)
                    M = torch.stack([Mr_ * scale_, Mi_ * scale_], dim=1)
                if tone_guard and tone_guard > 0:
                    M = apply_tone_guard(M, X, guard=tone_guard, thresh=tone_thresh)

                Mr, Mi = M[:, 0].float(), M[:, 1].float()
                Xr, Xi = X.real.float(), X.imag.float()
                Y = torch.complex(Mr * Xr - Mi * Xi, Mr * Xi + Mi * Xr)
                y = istft_from(Y, win, length=seg.shape[-1], n_fft=n_fft, hop=hop)

            if wiener_kappa and wiener_kappa > 0:
                y = wiener_refine(seg, y, win, n_fft=n_fft, hop=hop, kappa=wiener_kappa)

            y = strength * y + (1.0 - strength) * seg
            y[:, :fade_len] *= fade_in
            y[:, -fade_len:] *= fade_out
            chunks.append(y)

        pos += hop_len

    out = torch.zeros(C, (len(chunks) - 1) * hop_len + chunks[-1].shape[1], device=device)
    p = 0
    for ch in chunks:
        L = ch.shape[1]
        out[:, p:p + L] += ch
        p += hop_len
    return out[:, :T].cpu()


# ---------- Checkpoint loading ----------
def load_checkpoint(ckpt_path: str, device: str, prefer_ema: bool = True) -> tuple[ComplexUNet, int]:
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state) if isinstance(state, dict) else state
    base = None
    if isinstance(state, dict):
        base = state.get("args", {}).get("model", {}).get("base", None)
    if base is None:
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and k.endswith("enc1.net.0.conv.weight") and v.ndim == 4:
                base = int(v.shape[0]);
                break
    base = int(base) if base is not None else 48
    net = ComplexUNet(base=base).to(device)

    if prefer_ema and isinstance(state, dict) and state.get("ema", None) is not None:
        ema_sd = {k: v for k, v in state["ema"].items() if isinstance(v, torch.Tensor)}
        try:
            net.load_state_dict(ema_sd, strict=False)
        except Exception:
            net.load_state_dict(sd, strict=False)
    else:
        net.load_state_dict(sd, strict=False)
    net.eval()
    return net, base


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Denoise audio with ComplexUNet (chunked OLA, tone-guard, Wiener)")
    ap.add_argument("--ckpt", required=True, help="path to .pt (uses EMA if present)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--infile", type=str)
    group.add_argument("--in-dir", type=str)
    ap.add_argument("--outfile", type=str)
    ap.add_argument("--out-dir", type=str)
    ap.add_argument("--suffix", type=str, default="_denoised")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--strength", type=float, default=0.95)
    ap.add_argument("--wiener", type=float, default=1.5)
    ap.add_argument("--mask-limit", type=float, default=1.05)
    ap.add_argument("--tone-guard", type=float, default=0.6)
    ap.add_argument("--tone-thresh", type=float, default=0.45)
    ap.add_argument("--amp", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--chunk-sec", type=float, default=20.0)
    ap.add_argument("--overlap-sec", type=float, default=0.5)
    ap.add_argument("--no-ema", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    net, base = load_checkpoint(a.ckpt, device=device, prefer_ema=(not a.no_ema))
    if a.channels_last and device.startswith("cuda"):
        net = net.to(memory_format=torch.channels_last)

    targets = []
    if a.infile:
        if not a.outfile:
            raise ValueError("--outfile required with --infile")
        targets = [(Path(a.infile), Path(a.outfile))]
    else:
        in_dir = Path(a.in_dir)
        out_dir = Path(a.out_dir) if a.out_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(in_dir.iterdir()):
            if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"} and p.is_file():
                targets.append((p, out_dir / f"{p.stem}{a.suffix}.wav"))

    for inp, outp in targets:
        x, sr_in = sf.read(str(inp), always_2d=True)
        x = x.astype(np.float32)
        if sr_in != a.sr:
            x = resample_if_needed(x, sr_in, a.sr)
            sr_in = a.sr
        wav = torch.from_numpy(x).t().contiguous()  # (C,T)

        y = process_waveform(
            net=net, audio=wav, sr=sr_in, device=device,
            n_fft=a.n_fft, hop=a.hop,
            mask_limit=a.mask_limit, strength=a.strength,
            wiener_kappa=a.wiener, tone_guard=a.tone_guard, tone_thresh=a.tone_thresh,
            amp=a.amp, channels_last=a.channels_last,
            chunk_sec=a.chunk_sec, overlap_sec=a.overlap_sec,
        )
        outp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(outp), y.t().numpy(), sr_in)
        print(f"[done] {inp.name} -> {outp}")


if __name__ == "__main__":
    main()
