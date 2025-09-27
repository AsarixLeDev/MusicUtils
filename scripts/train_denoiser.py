#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================
# Train Complex-UNet denoiser (fast + robust)
# ================================================

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import inspect
from pathlib import Path
from typing import Dict, Any, List

# --- make 'soundrestorer' importable when running as: python scripts/train_denoiser.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ---------- perf switches (Ampere/Hopper/Blackwell friendly) ----------
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import torch.nn as nn
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
    category=UserWarning,
)

# project imports
from soundrestorer.models.denoiser_net import ComplexUNet
from soundrestorer.losses.mrstft import MultiResSTFTLoss
from soundrestorer.data.dataset import DenoiseDataset, DenoiseConfig
from soundrestorer.train.config import load_and_prepare  # YAML + run dirs helper

# =========== global switches ===========
USE_CHANNELS_LAST = True  # ok for (B,2,F,T) tensors

def seed_worker(worker_id: int):
    import random, numpy as np, torch
    info = torch.utils.data.get_worker_info()
    # base this worker's seed on PyTorch's initial_seed
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    # also reseed the dataset's per-worker RNG copy
    try:
        ds_local = info.dataset
        if hasattr(ds_local, "_rng"):
            ds_local._rng = np.random.RandomState(seed)
    except Exception:
        pass

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_wav(path: str, audio: torch.Tensor, sr: int):
    """
    audio: (T,), (1,T), or (C,T) on CPU or CUDA. Will clamp to [-1,1] and save as float32 WAV.
    """
    a = audio.detach()
    if a.is_cuda:
        a = a.cpu()
    # ensure (C,T)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    elif a.dim() == 2 and a.shape[0] == 1:
        pass
    elif a.dim() == 2 and a.shape[0] > 1:
        pass
    elif a.dim() == 3:
        # (B,C,T) -> take first sample
        a = a[0]
    else:
        a = a.view(1, -1)

    a = torch.clamp(a.float(), -1.0, 1.0).contiguous()

    # prefer soundfile; fall back to torchaudio
    try:
        import soundfile as sf
        sf.write(path, a.transpose(0,1).numpy(), sr, subtype="PCM_16")  # or "FLOAT"
    except Exception:
        import torchaudio
        torchaudio.save(path, a, sample_rate=sr, encoding="PCM_S", bits_per_sample=16)


# =========== small DSP helpers (self-contained) ===========
@torch.no_grad()
def si_sdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if y.dim() == 3: y = y.mean(dim=1); x = x.mean(dim=1)
    xz = x - x.mean(dim=-1, keepdim=True)
    yz = y - y.mean(dim=-1, keepdim=True)
    s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
         (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
    e = yz - s
    num = torch.sum(s ** 2, dim=-1)
    den = torch.sum(e ** 2, dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)

@torch.no_grad()
def si_sdr_db_best_shift(y: torch.Tensor, x: torch.Tensor, shifts=(0, 1, 2, 4, 8)) -> tuple[float, int]:
    if y.dim() == 3: y = y.mean(dim=1); x = x.mean(dim=1)
    B, T = y.shape
    best = -1e9; best_k = 0
    for k in sorted(set(list(shifts) + [-s for s in shifts if s != 0])):
        if k == 0:  yk = y
        elif k > 0: yk = torch.nn.functional.pad(y, (k, 0))[:, :T]
        else:       yk = torch.nn.functional.pad(y, (0, -k))[:, -k:T - k]
        val = si_sdr_db(yk, x).mean().item()
        if val > best: best, best_k = val, k
    return best, best_k

def stft_phase_cosine(
    y: torch.Tensor, x: torch.Tensor, win: torch.Tensor,
    n_fft: int, hop: int, eps: float = 1e-8, mag_weight: bool = True
) -> torch.Tensor:
    """
    Phase loss ~ mean(1 - cos Δφ) between STFTs of y and x.
    If mag_weight=True, weights by |X| to de-emphasize silent bins.
    Accepts (B,T) or (B,C,T) (collapses channels to mono).
    """
    Y = _stft_any(y, n_fft, hop, win)
    X = _stft_any(x, n_fft, hop, win)

    Yr, Yi = Y.real, Y.imag
    Xr, Xi = X.real, X.imag
    dot   = Yr * Xr + Yi * Xi
    denom = (Y.abs() * X.abs() + eps)
    cos_dphi = dot / denom  # cos(Δφ)

    if mag_weight:
        w = X.abs()
        loss = (w * (1.0 - cos_dphi)).sum() / (w.sum() + eps)
    else:
        loss = (1.0 - cos_dphi).mean()
    return loss


def _stft_any(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    """
    x: (T) or (B,T) or (B,C,T) -> returns complex STFT (B,F,T)
    Collapses multi-channel to mono for stability.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1,T)
    elif x.dim() == 3:
        x = x.mean(dim=1)  # (B,C,T) -> (B,T)
    elif x.dim() != 2:
        raise RuntimeError(f"STFT expects 1D/2D/3D, got shape {tuple(x.shape)}")
    return torch.stft(
        x.float(), n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True
    )  # (B,F,T)


def log_mel_L1(
    y: torch.Tensor,
    x: torch.Tensor,
    win: torch.Tensor,
    sr: int,
    n_mels: int = 64,
    n_fft: int | None = None,
    hop: int | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    assert n_fft is not None and hop is not None, "n_fft/hop are required"
    Y = _stft_any(y, n_fft, hop, win).abs()  # (B,F,T)
    X = _stft_any(x, n_fft, hop, win).abs()
    fb = _mel_filterbank(sr, n_fft, n_mels, device=Y.device)  # (M,F)
    Ym = torch.matmul(fb, Y)  # (B,M,T)
    Xm = torch.matmul(fb, X)
    return torch.mean(torch.abs(torch.log10(Ym + eps) - torch.log10(Xm + eps)))


def highband_mag_L1(
    y: torch.Tensor,
    x: torch.Tensor,
    win: torch.Tensor,
    sr: int,
    cutoff_khz: float = 8.0,
    n_fft: int | None = None,
    hop: int | None = None,
    use_log: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    L1 in the high band only, with correct bin mapping and log-stabilized magnitude.
    Works on (B,T) or (B,1,T) or (B,C,T) (collapses to mono internally).
    """
    assert n_fft is not None and hop is not None, "n_fft/hop are required"
    Y = _stft_any(y, n_fft, hop, win).abs()  # (B,F,T)
    X = _stft_any(x, n_fft, hop, win).abs()  # (B,F,T)

    F = n_fft // 2 + 1
    freqs = torch.arange(F, device=Y.device, dtype=Y.dtype) * (float(sr) / float(n_fft))
    cutoff_hz = float(cutoff_khz) * 1000.0
    hi_mask = freqs >= cutoff_hz
    if not torch.any(hi_mask):
        return Y.new_zeros(())

    Yh = Y[:, hi_mask, :]
    Xh = X[:, hi_mask, :]

    if use_log:
        diff = torch.abs(torch.log1p(Yh + eps) - torch.log1p(Xh + eps))
    else:
        diff = torch.abs(Yh - Xh)
    return diff.mean()


def hann_window(n_fft, device):
    w = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)
    return torch.sqrt(torch.clamp(w, min=0))


def stft_pair(wave: torch.Tensor, win: torch.Tensor, n_fft: int, hop: int):
    """
    wave: (B,T) or (B,1,T) -> returns (complex X[B,F,T], RI[B,2,F,T])
    STFT is fp32 with autocast disabled; explicit win_length and center flags.
    """
    # collapse to mono (B,T), enforce contiguous float32
    if wave.dim() == 3:
        wave = wave.mean(dim=1)  # (B,T)
    elif wave.dim() == 2:
        pass
    else:
        raise RuntimeError(f"stft_pair expects (B,T) or (B,1,T), got {tuple(wave.shape)}")
    wave = wave.contiguous().to(dtype=torch.float32)

    # window must be contiguous float32 on same device
    win = win.contiguous().to(dtype=torch.float32, device=wave.device)

    device_type = "cuda" if wave.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=False):
        X = torch.stft(
            wave, n_fft=n_fft, hop_length=hop,
            win_length=n_fft,                # explicit
            window=win, center=True,         # explicit
            return_complex=True, normalized=False
        )  # (B,F,T) complex
        Xri = torch.stack([X.real, X.imag], dim=1).contiguous()  # (B,2,F,T)
    return X, Xri


def istft_from(Xhat: torch.Tensor, win: torch.Tensor, length: int, n_fft: int, hop: int):
    """
    ISTFT is fp32 with autocast disabled; explicit win_length and center flags.
    """
    win = win.contiguous().to(dtype=torch.float32, device=Xhat.device)
    device_type = "cuda" if Xhat.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=False):
        # make sure complex is complex64 for numerical stability
        Xhat = Xhat.to(torch.complex64)
        y = torch.istft(
            Xhat, n_fft=n_fft, hop_length=hop,
            win_length=n_fft,               # explicit
            window=win, center=True,        # explicit
            length=length, normalized=False
        )  # (B,T)
    return y


def try_bias_head_to_identity(net: nn.Module):
    """
    If the last conv that produces (Re, Im) is accessible, bias it so:
      Re bias ~ 0 (delta1) or 1 (plain), Im bias ~ 0.
    We default to delta1 mode here (bias=0). Call this once after model init.
    """
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.out_channels == 2:
            with torch.no_grad():
                if mod.bias is not None:
                    mod.bias.zero_()  # delta1 = start near 0; for 'plain', set mod.bias[0]=1.0
            print(f"[init] head bias zeroed at {name}")
            break


# ---------- extra losses (pure torch, no torchaudio required) ----------
def _hz_to_mel(f):  # HTK
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None, device=None):
    fmax = float(sr) / 2.0 if fmax is None else float(fmax)
    m_min, m_max = _hz_to_mel(torch.tensor(fmin)), _hz_to_mel(torch.tensor(fmax))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    bins = torch.floor((n_fft + 1) * f_pts / float(sr)).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left: center] = torch.linspace(0, 1, center - left)
        if right > center:
            fb[i, center: right] = torch.linspace(1, 0, right - center)
    return fb.to(device=device, dtype=torch.float32)


def si_sdr_ratio_loss(y: torch.Tensor, x: torch.Tensor, min_db: float = -10.0, eps: float = 1e-8) -> torch.Tensor:
    y = _as_batch_mono(y)
    x = _as_batch_mono(x)
    xz = x - x.mean(dim=-1, keepdim=True)
    yz = y - y.mean(dim=-1, keepdim=True)
    s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
         (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
    e = yz - s
    si_sdr = 10 * torch.log10(torch.sum(s**2, dim=-1) / (torch.sum(e**2, dim=-1) + eps) + eps)
    denom = max(abs(float(min_db)), 1e-6)
    loss = torch.clamp((min_db - si_sdr) / denom, min=0.0)
    return torch.mean(loss)


# =========== EMA ===========
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.device = next(iter(self.shadow.values())).device

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    v.copy_(self.shadow[k])


# =========== Sampler + Mining ===========
class MutableWeightedSampler(torch.utils.data.Sampler[int]):
    def __init__(self, weights: torch.Tensor, num_samples: int, replacement: bool = True):
        self.weights = weights.clone().to(dtype=torch.double)
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)

    def set_weights(self, weights: torch.Tensor):
        self.weights = weights.clone().to(dtype=torch.double)

    def __iter__(self):
        idx = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


class HardMiner:
    def __init__(self, ema=0.9, top_frac=0.2, boost=3.0, baseline=1.0):
        self.ema = float(ema)
        self.top_frac = float(top_frac)
        self.boost = float(boost)
        self.base = float(baseline)
        self.stats: Dict[str, float] = {}

    def _flatten_ids(self, ids, target_len: int | None = None) -> list[str] | None:
        if ids is None:
            return None
        # meta["id"] might be a single str, list[str], list[list[str]], or tuple variants
        if isinstance(ids, str):
            return [ids] * (target_len if target_len is not None else 1)
        flat = []
        if isinstance(ids, dict):
            # extremely defensive; try common key
            ids = ids.get("id", list(ids.values()))
        if isinstance(ids, (list, tuple)):
            for it in ids:
                if isinstance(it, (list, tuple)):
                    flat.extend(it)
                else:
                    flat.append(it)
        else:
            try:
                flat = [str(ids)]
            except Exception:
                return None
        flat = [str(x) for x in flat]
        return flat

    def update_batch(self, ids, per_sample_losses: torch.Tensor | list | tuple):
        vals_t = torch.as_tensor(per_sample_losses)  # handles tensor or list/tuple
        vals_t = vals_t.detach().reshape(-1).float().cpu()
        vals = vals_t.tolist()

        ids_flat = self._flatten_ids(ids, target_len=len(vals))
        if not ids_flat:
            return

        for _id, v in zip(ids_flat, vals):
            try:
                vf = float(v)
            except Exception:
                continue
            if not math.isfinite(vf):
                continue
            if _id in self.stats:
                self.stats[_id] = self.ema * self.stats[_id] + (1.0 - self.ema) * vf
            else:
                self.stats[_id] = vf

    def make_weights(self, dataset: DenoiseDataset) -> torch.Tensor:
        ids = [dataset._rec_id_from_record(r) for r in dataset.records]
        arr = [self.stats.get(i, 0.0) for i in ids]
        if self.top_frac > 0:
            k = max(1, int(len(arr) * self.top_frac))
            thr = sorted(arr)[-k]
        else:
            thr = float("inf")
        w = [self.base * self.boost if v >= thr else self.base for v in arr]
        return torch.tensor(w, dtype=torch.double)


# =========== CUDA prefetcher ===========
class CUDAPrefetcher:
    def __init__(self, loader, device, channels_last: bool):
        self.loader = iter(loader)
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream() if device.startswith("cuda") else None
        self.next_batch = None
        self._preload()

    def _to(self, t):
        if t is None:
            return None
        # if already a Tensor, avoid re-wrapping via as_tensor
        if isinstance(t, torch.Tensor):
            return t.to(self.device, non_blocking=True)
        return torch.as_tensor(t).to(self.device, non_blocking=True)

    def _preload(self):
        if self.stream is None:
            try:
                self.next_batch = next(self.loader)
            except StopIteration:
                self.next_batch = None
            return

        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                noisy, clean, extra, meta = batch
            else:
                # back-compat: 3-tuple
                noisy, clean, extra = batch
                meta = None
            noisy = self._to(noisy)
            clean = self._to(clean)
            extra = self._to(extra) if isinstance(extra, torch.Tensor) else extra
            self.next_batch = (noisy, clean, extra, meta)

    def next(self):
        if self.stream is None:
            b = self.next_batch
            if b is None:
                return None
            self._preload()
            return b
        torch.cuda.current_stream().wait_stream(self.stream)
        b = self.next_batch
        if b is None:
            return None
        self._preload()
        return b


# =========== curriculum helpers ===========
def pick_curriculum_stage(curr_cfg: dict, epoch: int, default_min: float, default_max: float, default_p: float):
    if not curr_cfg or not curr_cfg.get("enable", False):
        return default_min, default_max, default_p
    stages = sorted(curr_cfg.get("stages", []), key=lambda s: int(s.get("until", 10**9)))
    for st in stages:
        if epoch <= int(st.get("until", 10**9)):
            return (
                float(st.get("snr_min", default_min)),
                float(st.get("snr_max", default_max)),
                float(st.get("use_ext_noise_p", default_p)),
            )
    last = stages[-1] if stages else {}
    return (
        float(last.get("snr_min", default_min)),
        float(last.get("snr_max", default_max)),
        float(last.get("use_ext_noise_p", default_p)),
    )


def sisdr_target_for_epoch(cfg: dict, epoch: int) -> float:
    sc = cfg.get("sisdr_curriculum", {})
    if not sc.get("enable", False):
        return float(cfg["loss"].get("sisdr_min_db", 0.0))
    start_db = float(sc.get("start_db", 0.0))
    end_db = float(sc.get("end_db", 10.0))
    end_ep = int(sc.get("end_epoch", 20))
    t = 0.0 if end_ep <= 1 else min(1.0, max(0.0, (epoch - 1) / (end_ep - 1)))
    return start_db + (end_db - start_db) * t


# =========== plotting (optional) ===========
class LivePlot:
    def __init__(self, enable: bool, backend: str | None, outfile: str | None):
        self.enable = False
        if enable:
            try:
                import matplotlib

                if backend:
                    matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                self.plt = plt
                self.enable = True
            except Exception:
                self.enable = False
        self.outfile = outfile
        if self.enable:
            self.fig, (self.ax1, self.ax2) = self.plt.subplots(1, 2, figsize=(12, 4))
            self.tr_x, self.tr_y = [], []
            self.va_x, self.va_y = [], []
            self.ax1.set_title("Current epoch")
            self.ax1.set_xlabel("step in epoch")
            self.ax1.set_ylabel("loss")
            self.ax2.set_title("Whole session")
            self.ax2.set_xlabel("global step / epoch")
            self.ax2.set_ylabel("loss")
            self.plt.tight_layout()

    def add_train(self, x, y):
        if not self.enable:
            return
        self.tr_x.append(x)
        self.tr_y.append(y)
        if len(self.tr_x) > 1 and x % 10 == 0:
            self.ax1.cla()
            self.ax1.set_title("Current epoch")
            self.ax1.set_xlabel("step in epoch")
            self.ax1.set_ylabel("loss")
            self.ax1.plot(self.tr_x, self.tr_y)
            self.plt.pause(0.001)

    def add_val(self, epoch, y):
        if not self.enable:
            return
        self.va_x.append(epoch)
        self.va_y.append(y)
        self.ax2.cla()
        self.ax2.set_title("Whole session")
        self.ax2.set_xlabel("global step / epoch")
        self.ax2.set_ylabel("loss")
        self.ax2.plot(self.tr_x, self.tr_y, label="train (EMA)")
        self.ax2.scatter(self.va_x, self.va_y, s=10, label="val (per epoch)")
        self.ax2.legend()
        self.plt.pause(0.001)
        if self.outfile:
            try:
                self.fig.savefig(self.outfile, dpi=120)
            except Exception:
                pass


# =========== loader builder ===========
def build_loader(manifest: str, ds_cfg: Dict[str, Any], train: bool):
    cfg = DenoiseConfig(
        sample_rate=ds_cfg["sr"],
        crop_seconds=ds_cfg["crop"],
        mono=True,
        seed=ds_cfg.get("seed", 0),
        enable_cache=not ds_cfg.get("no_cache", False),
        cache_gb=float(ds_cfg.get("cache_gb", 2.0)),
        snr_db_min=float(ds_cfg.get("snr_min", -5.0)),
        snr_db_max=float(ds_cfg.get("snr_max", 20.0)),
        use_external_noise_prob=float(ds_cfg.get("use_ext_noise_p", 0.7)),
        add_synthetic_noise_prob=float(ds_cfg.get("add_synth_noise_p", 1.0)),
        min_clean_rms_db=float(ds_cfg.get("min_clean_rms_db", -55.0)),
        max_retries=int(ds_cfg.get("max_retries", 6)),
        out_peak=float(ds_cfg.get("out_peak", 0.98)),
    )
    ds = DenoiseDataset(manifest, cfg, return_meta=False)

    batch = int(ds_cfg.get("batch", 8))
    workers = int(ds_cfg.get("workers", 4))
    pin = True
    persistent = workers > 0

    sampler = None
    if train and ds_cfg.get("hard_mining", {}).get("enable", False):
        num_samples = int(math.ceil(len(ds) / batch)) * batch
        weights = torch.ones(len(ds), dtype=torch.double)
        sampler = MutableWeightedSampler(weights, num_samples, replacement=True)

    prefetch = int(ds_cfg.get("prefetch_factor", 4))
    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch,
        shuffle=(train and sampler is None),
        drop_last=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        sampler=sampler,
    )
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch

    gen = torch.Generator()
    gen.manual_seed(int(ds_cfg.get("seed", 0)))  # allow config seeding if you add data.seed

    ld = torch.utils.data.DataLoader(
        **loader_kwargs,
        worker_init_fn=seed_worker,
        generator=gen,
    )
    return ds, ld, sampler


# =========== CLI ===========
def parse_args():
    ap = argparse.ArgumentParser(description="Train Complex-UNet denoiser")
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--set", nargs="*", default=[], help="Overrides like key.sub=val key2=3")
    ap.add_argument("--live-plot", action="store_true")
    ap.add_argument("--mpl-backend", default=None)
    ap.add_argument("--plot-file", default=None)
    ap.add_argument("--resume", default="", help="checkpoint .pt or dir (latest epoch_XXX.pt)")
    return ap.parse_args()


# =========== resume utils ===========
def find_latest_epoch_ckpt(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    cands = sorted([p for p in folder.glob("epoch_*.pt") if p.is_file()])
    return cands[-1] if cands else None


# =========== debug helpers (STFT sanity) ===========
def _as_batch_mono(x: torch.Tensor) -> torch.Tensor:
    # (T) -> (1,T), (B,T) -> (B,T), (B,C,T) -> (B,T)
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.mean(dim=1)
    raise RuntimeError(f"Expected 1D/2D/3D waveform, got {tuple(x.shape)}")


@torch.no_grad()
def stft_istft_sanity(wave: torch.Tensor, n_fft: int, hop: int, center: bool = True, periodic: bool = True,
                      normalized: bool = False) -> float:
    """
    Checks STFT->ISTFT round-trip with given flags. Returns SI-SDR(dB) between
    the reconstructed signal and the input. Safely handles center=False via reflect padding.
    """
    import torch.nn.functional as F

    wave = _as_batch_mono(wave)  # (B,T)
    win = torch.hann_window(n_fft, periodic=periodic, device=wave.device, dtype=torch.float32)

    if center:
        X = torch.stft(wave.float(), n_fft=n_fft, hop_length=hop, window=win,
                       center=True, return_complex=True, normalized=normalized)  # (B,F,T)
        y = torch.istft(X, n_fft=n_fft, hop_length=hop, window=win,
                        length=wave.shape[-1], center=True, normalized=normalized)
        ref = wave
    else:
        pad = n_fft // 2
        wavep = F.pad(wave, (pad, pad), mode="reflect")
        X = torch.stft(wavep.float(), n_fft=n_fft, hop_length=hop, window=win,
                       center=False, return_complex=True, normalized=normalized)
        yp = torch.istft(X, n_fft=n_fft, hop_length=hop, window=win,
                         center=False, normalized=normalized)
        y = yp[..., pad:-pad]
        ref = wave

    def _si_db(y, x, eps=1e-8):
        xz = x - x.mean(dim=-1, keepdim=True)
        yz = y - y.mean(dim=-1, keepdim=True)
        s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
             (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
        e = yz - s
        return 10 * torch.log10(torch.sum(s ** 2, dim=-1) / (torch.sum(e ** 2, dim=-1) + eps) + eps)

    return _si_db(y, ref).mean().item()


@torch.no_grad()
def si_sdr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = _as_batch_mono(y)
    x = _as_batch_mono(x)
    xz = x - x.mean(dim=-1, keepdim=True)
    yz = y - y.mean(dim=-1, keepdim=True)
    s = (torch.sum(yz * xz, dim=-1, keepdim=True) /
         (torch.sum(xz * xz, dim=-1, keepdim=True) + eps)) * xz
    e = yz - s
    num = torch.sum(s ** 2, dim=-1)
    den = torch.sum(e ** 2, dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)


# =========== main ===========
def main():
    args = parse_args()

    cfg, paths = load_and_prepare(args.config, overrides=args.set)  # paths has run dirs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # run folders
    ckpt_dir = Path(paths.get("checkpoints", cfg.get("paths", {}).get("checkpoints", "checkpoints/denoiser")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lambda_sisdr = cfg['loss'].get('lambda_sisdr', 0.0)

    # summary
    print("\n===== training summary =====")
    print(f"Device: {device} | AMP: {cfg['runtime'].get('amp', 'bfloat16')}")
    print(f"SR: {cfg['data']['sr']} | crop: {cfg['data']['crop']}s | batch: {cfg['data']['batch']}")
    print(f"Workers: {cfg['data']['workers']} | prefetch: {cfg['data'].get('prefetch_factor', 4)}")
    print(f"Model: ComplexUNet(base={cfg['model']['base']}) | channels_last: {cfg['runtime'].get('channels_last', True)}")
    # datasets
    tr_ds, tr_ld, tr_sampler = build_loader(cfg["data"]["train_manifest"], cfg["data"], train=True)
    va_ds, va_ld, _ = build_loader(cfg["data"]["val_manifest"], cfg["data"], train=False)
    print(f"Train items: {len(tr_ds.records)} | steps/epoch: {len(tr_ld)}")
    print(f"Val items:   {len(va_ds.records)}")
    print(
        f"Loss weights: sisdr={lambda_sisdr} nf={cfg['loss'].get('lambda_nf', 0.0)} "
        f"mel={cfg['loss'].get('lambda_mel', 0.0)} hi={cfg['loss'].get('lambda_hi', 0.0)}"
    )
    print(
        f"Identity: p_clean={cfg['data'].get('p_clean', 0.0):.3f} "
        f"λ_idmask={cfg['loss'].get('lambda_idmask', 0.0)} λ_idwav={cfg['loss'].get('lambda_idwav', 0.0)}"
    )
    print(f"Mask limit: {cfg['loss'].get('mask_limit', 1.0)}")
    print(f"Checkpoints: {ckpt_dir}")
    print("===========================\n")

    # right after you build the loaders, before training:
    with torch.no_grad():
        it = iter(tr_ld)
        bad = 0
        for _ in range(8):  # a handful of batches is enough
            noisy, clean, _, _ = next(it)
            # use the SAME shapes stft_pair used
            nm = noisy.mean(dim=1) if noisy.dim() == 3 else noisy
            sm = clean.mean(dim=1) if clean.dim() == 3 else clean
            si = si_sdr_db(nm, sm).mean().item()
            print(f"[pre-train loader SI] noisy_vs_clean={si:.2f} dB")
            if si < 0:
                bad += 1
        if bad:
            print(f"[alert] {bad} of these batches had SI<0 dB → alignment still broken.")

    # model
    net = ComplexUNet(base=int(cfg["model"]["base"])).to(device)
    if USE_CHANNELS_LAST and device.startswith("cuda") and cfg["runtime"].get("channels_last", True):
        net = net.to(memory_format=torch.channels_last)

    try_bias_head_to_identity(net)

    # AMP flags
    amp_cfg = str(cfg["runtime"].get("amp", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_cfg == "bfloat16" else torch.float16
    amp_enabled = device.startswith("cuda") and amp_cfg != "float32"

    # optimizer
    use_fp16_math = (amp_enabled and amp_dtype is torch.float16)
    opt_kwargs = dict(
        lr=float(cfg["optim"]["lr"]),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(cfg["optim"].get("wd", 0.0)),
    )

    # Detect support in this PyTorch build
    sig = inspect.signature(torch.optim.AdamW.__init__).parameters
    supports_fused = "fused" in sig
    supports_foreach = "foreach" in sig
    on_cuda = device.startswith("cuda")

    # Prefer fused on CUDA when available; disable fused for fp16 AMP to avoid older-PT asserts
    prefer_fused = (on_cuda and supports_fused and not use_fp16_math)

    try:
        if prefer_fused:
            # fused only
            opt = torch.optim.AdamW(net.parameters(), fused=True, foreach=False, **opt_kwargs)
            print("[optim] AdamW fused=True")
        elif supports_foreach:
            # foreach only
            opt = torch.optim.AdamW(net.parameters(), foreach=True, **opt_kwargs)
            print("[optim] AdamW foreach=True")
        else:
            # plain
            opt = torch.optim.AdamW(net.parameters(), **opt_kwargs)
            print("[optim] AdamW plain")
    except (TypeError, RuntimeError):
        # Fallback chain if this build/runtime rejects the chosen flag
        try:
            if supports_foreach:
                opt = torch.optim.AdamW(net.parameters(), foreach=True, **opt_kwargs)
                print("[optim] AdamW fallback foreach=True")
            else:
                opt = torch.optim.AdamW(net.parameters(), **opt_kwargs)
                print("[optim] AdamW fallback plain")
        except (TypeError, RuntimeError):
            opt = torch.optim.AdamW(net.parameters(), **opt_kwargs)
            print("[optim] AdamW final fallback plain")

    # scheduler (linear warmup → cosine)
    total_epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(1, len(tr_ld))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(cfg["optim"].get("warmup_steps", 0))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay to 10% floor
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # AMP scaler (new API first, fallback to old if needed)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_math)
    except Exception:
        from torch.cuda.amp import GradScaler as _OldGradScaler

        scaler = _OldGradScaler(enabled=use_fp16_math)

    # optional compile
    if bool(cfg["runtime"].get("compile", False)) and hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead")
            print("[compile] torch.compile active")
        except Exception as e:
            print(f"[compile] disabled: {e}")

    # EMA
    ema = EMA(net, decay=float(cfg["train"].get("ema", 0.999))) if float(cfg["train"].get("ema", 0.0)) > 0 else None

    # MR-STFT loss instance
    mrstft = MultiResSTFTLoss().to(device)

    # windows (shared)
    n_fft = int(cfg["inference_defaults"]["n_fft"])
    hop = int(cfg["inference_defaults"]["hop"])
    win = hann_window(n_fft, device)

    # curriculum + mining setup
    miner_cfg = cfg.get("hard_mining", {})
    use_hard = bool(miner_cfg.get("enable", False))
    miner = (
        HardMiner(
            ema=float(miner_cfg.get("ema", 0.9)),
            top_frac=float(miner_cfg.get("top_frac", 0.2)),
            boost=float(miner_cfg.get("boost", 3.0)),
            baseline=float(miner_cfg.get("baseline", 1.0)),
        )
        if use_hard
        else None
    )
    start_hm_epoch = int(miner_cfg.get("start_epoch", 3))

    # live plotting
    live = LivePlot(args.live_plot, args.mpl_backend, args.plot_file)

    # resume (optional)
    start_epoch = 1
    global_step = 0
    if args.resume:
        p = Path(args.resume)
        ckpt_path = p if p.suffix == ".pt" else find_latest_epoch_ckpt(p)
        if ckpt_path and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state.get("model", state)
            missing, unexpected = net.load_state_dict(sd, strict=False)
            print(f"[resume] loaded {ckpt_path.name}  missing={len(missing)} unexpected={len(unexpected)}")
            if "opt" in state:
                try:
                    opt.load_state_dict(state["opt"])
                    print("[resume] optimizer ok")
                except Exception as e:
                    print(f"[resume] optimizer skip: {e}")
            if "sched" in state:
                try:
                    scheduler.load_state_dict(state["sched"])
                except Exception:
                    pass
            start_epoch = state.get("epoch", start_epoch)
            global_step = state.get("global_step", (start_epoch - 1) * steps_per_epoch)
            if ema and "ema" in state:
                try:
                    ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
                    print("[resume] EMA ok")
                except Exception:
                    pass
        else:
            print(f"[resume] path not found: {args.resume}")

    noisy, _, _, _ = next(iter(va_ld))
    noisy = noisy.to(device)
    print("center=True sanity:", stft_istft_sanity(noisy, n_fft, hop, center=True, periodic=True))

    @torch.no_grad()
    def dataset_pairing_report(ds, n=128):
        import numpy as np, random, torch
        ok_threshold_db = getattr(ds.cfg, "snr_db_min", 0.0)
        idxs = random.sample(range(len(ds)), k=min(n, len(ds)))
        vals = []
        for i in idxs:
            noisy_f, clean_f, _, _ = ds[i]
            nmono = noisy_f.mean(axis=0) if noisy_f.ndim == 2 else noisy_f
            cmono = clean_f.mean(axis=0) if clean_f.ndim == 2 else clean_f
            si = float(si_sdr_db(torch.from_numpy(nmono)[None, :],
                                 torch.from_numpy(cmono)[None, :]).mean().item())
            vals.append((i, si))
        ok = [(i, v) for (i, v) in vals if v > ok_threshold_db - 1e-6]
        bad = [(i, v) for (i, v) in vals if v <= ok_threshold_db - 1e-6]
        print(f"[pairing] ok={len(ok)} bad={len(bad)} of {len(idxs)}")
        if bad:
            print("worst 10 bad:", bad[:10])
        if ok:
            print("best 10 ok:", ok[-10:])

    # After build_loader(...)
    print("=== TRAIN DATA PAIRING ===")
    dataset_pairing_report(tr_ds, n=128)
    print("=== VAL DATA PAIRING ===")
    dataset_pairing_report(va_ds, n=128)

    # training knobs
    grad_accum = int(cfg["optim"].get("grad_accum", 1))
    loss_clip = float(cfg["loss"].get("train_loss_clip", 0.0))  # 0 disables
    channels_last_en = USE_CHANNELS_LAST and device.startswith("cuda") and cfg["runtime"].get("channels_last", True)
    use_prefetch = device.startswith("cuda") and bool(cfg["runtime"].get("cuda_prefetch", True))

    # --------- epochs ----------
    for epoch in tqdm(range(start_epoch, total_epochs + 1), desc="epochs", dynamic_ncols=True):
        net.train()
        # curriculum SNR for this epoch
        snr_min0 = float(cfg["data"].get("snr_min", -5.0))
        snr_max0 = float(cfg["data"].get("snr_max", 20.0))
        p_ext0 = float(cfg["data"].get("use_ext_noise_p", 0.7))
        snr_min_e, snr_max_e, p_ext_e = pick_curriculum_stage(cfg.get("curriculum", {}), epoch, snr_min0, snr_max0, p_ext0)
        tr_ds.set_snr_range(snr_min_e, snr_max_e)
        tr_ds.set_use_ext_noise_p(p_ext_e)

        tot = 0.0
        used = 0
        skipped = 0
        last_loss = None

        # iterator (prefetch vs plain)
        if use_prefetch:
            prefetch = CUDAPrefetcher(tr_ld, device, channels_last_en)
            batch = prefetch.next()
        else:
            it = iter(tr_ld)
            batch = next(it, None)

        # per-epoch progress bar over **batches**
        train_bar = tqdm(total=len(tr_ld), desc=f"train {epoch}", leave=False, dynamic_ncols=True)

        step_in_epoch = 0
        while batch is not None:
            # tuple may be 3 or 4 elems (with meta)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                noisy, clean, flags_or_none, meta = batch
                # meta is usually a dict of lists after default collate
                if isinstance(meta, dict):
                    ids = meta.get("id", None)
                elif isinstance(meta, (list, tuple)):
                    # rare: collate as list of dicts
                    ids = [m.get("id") if isinstance(m, dict) else m for m in meta]
                else:
                    ids = meta
            else:
                noisy, clean, flags_or_none = batch
                ids = None

            step_in_epoch += 1
            # print("dtypes:", noisy.dtype, win.dtype)

            with torch.autocast(
                device_type=("cuda" if device.startswith("cuda") else "cpu"),
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                # ---- 1) STFT in float32 OUTSIDE autocast ----
                # A) STFT (autocast disabled internally by stft_pair) → float32 complex
                Xn, Xn_ri = stft_pair(noisy, win, n_fft=n_fft, hop=hop)
                if channels_last_en:
                    Xn_ri = Xn_ri.contiguous(memory_format=torch.channels_last)

                # B) net forward under autocast
                with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=amp_dtype,
                                    enabled=amp_enabled):
                    Xn_ri_amp = Xn_ri.to(amp_dtype) if amp_enabled else Xn_ri
                    M = net(Xn_ri_amp)

                # C) head mapping & effective-mask clamp in float32
                Mr, Mi = M[:, 0].float(), M[:, 1].float()
                Xr, Xi = Xn.real.float(), Xn.imag.float()

                mask_variant = str(cfg["model"].get("mask_variant", "mag_sigm1")).lower()

                if mask_variant == "mag_sigm1":
                    Mag = 1.0 + (torch.sigmoid(Mr) - 0.5)  # ~[0.5, 1.5]
                    R, I = Mag, torch.zeros_like(Mag)
                elif mask_variant == "mag":
                    Mag = torch.sqrt(Mr ** 2 + Mi ** 2 + 1e-8)
                    R, I = Mag, torch.zeros_like(Mag)
                elif mask_variant == "plain":
                    R, I = Mr, Mi
                elif mask_variant == "delta1":
                    R, I = 1.0 + Mr, Mi
                elif mask_variant == "mag_delta1":
                    Mag = 1.0 + torch.sqrt(Mr ** 2 + Mi ** 2 + 1e-8)
                    R, I = Mag, torch.zeros_like(Mag)
                else:
                    raise ValueError(f"Unknown model.mask_variant={mask_variant!r}")

                mlim = float(cfg["loss"].get("mask_limit", 0.0))
                if mlim > 0:
                    mag_eff = torch.sqrt(R ** 2 + I ** 2 + 1e-8)
                    scale = torch.clamp(mlim / mag_eff, max=1.0)
                    R, I = R * scale, I * scale

                Xhat = torch.complex(R * Xr - I * Xi, R * Xi + I * Xr)

                # D) ISTFT (autocast disabled internally by istft_from)
                yhat = istft_from(Xhat, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)

                clean_m = _as_batch_mono(clean.float())  # (B,T)
                noisy_m = _as_batch_mono(noisy.float())  # (B,T)

                # === DEBUG AUDIO SAVE (first item of batch) ===
                if (global_step % 100) == 0 or global_step == 0:
                    out_root = "debug_audio/train"
                    _ensure_dir(out_root)

                    # grab first item (mono) for simplicity
                    n0 = noisy[0]  # (C,T) or (T)
                    c0 = clean[0]
                    y0 = yhat[0]

                    # optional: unity recon to verify transform
                    Xn_chk, _ = stft_pair(noisy, win, n_fft, hop)
                    y_unity = istft_from(Xn_chk, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)[0]

                    # make a friendly stem: epoch_step_id
                    if isinstance(meta, dict) and "id" in meta and len(meta["id"]) > 0:
                        stem = f"e{epoch:03d}_s{global_step}_id"
                        # meta['id'] is probably a list; take first
                        try:
                            id0 = str(meta["id"][0]).split("/")[-1].split("\\")[-1]
                            stem += f"_{id0[:40]}"
                        except Exception:
                            pass
                    else:
                        stem = f"e{epoch:03d}_s{global_step}"

                    sr = int(cfg["data"]["sr"])

                    save_wav(os.path.join(out_root, f"{stem}_noisy.wav"), n0, sr)
                    save_wav(os.path.join(out_root, f"{stem}_clean.wav"), c0, sr)
                    save_wav(os.path.join(out_root, f"{stem}_pred.wav"), y0, sr)
                    save_wav(os.path.join(out_root, f"{stem}_unity.wav"), y_unity, sr)

                # losses
                lambda_phase = float(cfg["loss"].get("lambda_phase", 0.0))
                phase = stft_phase_cosine(yhat, clean.float(), win, n_fft=n_fft,
                                          hop=hop) if lambda_phase > 0 else 0.0

                mr_out = mrstft(yhat, clean.float())
                mr = mr_out[0] if isinstance(mr_out, tuple) else mr_out
                l1 = torch.mean(torch.abs(yhat - clean_m))
                # SI-SDR (ratio) with epoch target + cap
                sisdr_target_db = sisdr_target_for_epoch(cfg, epoch)
                if float(cfg["loss"].get("lambda_sisdr", 0.0)) > 0:
                    si = si_sdr_ratio_loss(yhat, clean_m, min_db=sisdr_target_db) if lambda_sisdr > 0 else 0.0
                    si_db_val = si_sdr_db(yhat, clean_m).mean().item()
                    si_cap = float(cfg["loss"].get("sisdr_loss_cap", 2.0))
                    if isinstance(si, torch.Tensor):
                        si = torch.clamp(si, max=si_cap)
                else:
                    si = 0.0

                mel = (
                    log_mel_L1(
                        yhat,
                        clean.float(),
                        win,
                        sr=cfg["data"]["sr"],
                        n_mels=int(cfg["loss"].get("mel_bands", 64)),
                        n_fft=n_fft,
                        hop=hop,
                    )
                    if float(cfg["loss"].get("lambda_mel", 0.0)) > 0
                    else 0.0
                )
                hi = (
                    highband_mag_L1(
                        yhat,
                        clean.float(),
                        win,
                        sr=cfg["data"]["sr"],
                        cutoff_khz=float(cfg["loss"].get("hi_emph_khz", 8.0)),
                        n_fft=n_fft,
                        hop=hop,
                    )
                    if float(cfg["loss"].get("lambda_hi", 0.0)) > 0
                    else 0.0
                )

                loss = (
                    mr
                    + 0.5 * l1
                    + float(cfg["loss"].get("lambda_sisdr", 0.0))
                    * (si if isinstance(si, torch.Tensor) else torch.tensor(si, device=device))
                    + float(cfg["loss"].get("lambda_mel", 0.0))
                    * (mel if isinstance(mel, torch.Tensor) else torch.tensor(mel, device=device))
                    + float(cfg["loss"].get("lambda_hi", 0.0))
                    * (hi if isinstance(hi, torch.Tensor) else torch.tensor(hi, device=device))
                    + lambda_phase
                    * (phase if isinstance(phase, torch.Tensor) else torch.tensor(phase, device=device))
                )

                # ---- energy regularizer (keep output RMS near input RMS) ----
                lambda_energy = float(cfg["loss"].get("lambda_energy", 0.0))
                if lambda_energy > 0:
                    y_std = yhat.float().std(dim=-1).mean()
                    x_std = noisy_m.float().std(dim=-1).mean()
                    loss = loss + lambda_energy * ((y_std / (x_std + 1e-8)) - 1.0).pow(2)

                lambda_idmask = float(cfg["loss"].get("lambda_idmask", 0.0))
                if lambda_idmask > 0:
                    if mask_variant == "mag_sigm1":
                        reg = (R - 1.0).pow(2).mean()  # R is Mag here
                    elif mask_variant in ("mag", "mag_delta1"):
                        reg = (R - (1.0 if mask_variant == "mag" else 1.0)).pow(2).mean()
                    elif mask_variant == "plain":
                        reg = (Mr - 1.0).pow(2).mean() + (Mi - 0.0).pow(2).mean()
                    elif mask_variant == "delta1":
                        reg = (Mr - 0.0).pow(2).mean() + (Mi - 0.0).pow(2).mean()
                    else:
                        reg = torch.zeros((), device=yhat.device)
                    loss = loss + lambda_idmask * reg


                # per-sample proxy for miner
                if yhat.dim() == 3:
                    y_m = yhat.mean(dim=1)
                    c_m = clean.float().mean(dim=1)
                else:
                    y_m = yhat
                    c_m = clean.float()
                l1_per = torch.mean(torch.abs(y_m - c_m), dim=-1)  # (B,)

            # silent guard (skip bad batches)
            s = float(loss.detach().item())
            if (not math.isfinite(s)) or (loss_clip > 0 and s > loss_clip):
                skipped += 1
                train_bar.update(1)  # still advance the bar
                if step_in_epoch % 10 == 0 and last_loss is not None:
                    train_bar.set_postfix(
                        loss=f"{last_loss:.4f}", used=used, skip=skipped, lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )
                # next batch
                if use_prefetch:
                    batch = prefetch.next()
                else:
                    batch = next(it, None)
                continue

            # optimizer step (with accumulation)
            if scaler.is_enabled():
                scaler.scale(loss / grad_accum).backward()
                if (global_step + 1) % grad_accum == 0:
                    scaler.unscale_(opt)
                    if float(cfg["optim"].get("grad_clip", 0.0)) > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(cfg["optim"]["grad_clip"]))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                (loss / grad_accum).backward()
                if (global_step + 1) % grad_accum == 0:
                    if float(cfg["optim"].get("grad_clip", 0.0)) > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(cfg["optim"]["grad_clip"]))
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            scheduler.step()
            if ema:
                ema.update(net)
            if ids is not None and isinstance(ids, list) and miner is not None:
                miner.update_batch(ids, l1_per)

            global_step += 1
            # component debug (safe detach)
            if (global_step % 100) == 0 or global_step == 0:
                with torch.no_grad():
                    # 1) Baseline: how good is the mixture vs. clean on this batch?
                    siddr_noisy = si_sdr_db(noisy, clean).mean().item()

                    # 2) Unity mask via *your* STFT/ISTFT path — should match the baseline above
                    # (so we detect reconstruction alignment issues, not model issues)
                    y_unity = istft_from(Xn, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)
                    siddr_unity = si_sdr_db(y_unity, clean).mean().item()

                    # 3) Best small shift (to see if tiny offsets are the culprit)
                    best_noisy, k_n = si_sdr_db_best_shift(noisy, clean)
                    best_yhat, k_r = si_sdr_db_best_shift(yhat, clean)
                    si_db_val = si_sdr_db(yhat, clean.float()).mean().item()
                print(
                    f"[unity] step {global_step} -> noisy={siddr_noisy:.2f} dB | unity={siddr_unity:.2f} dB | restored={si_db_val:.2f} dB")
                print(f"[siddr] step {global_step} -> "
                      f"noisy={siddr_noisy:.2f} dB | unity={siddr_unity:.2f} dB | "
                      f"restored={si_sdr_db(yhat, clean).mean().item():.2f} dB | "
                      f"best noisy={best_noisy:.2f}@{k_n}, best restored={best_yhat:.2f}@{k_r}")
                print(f"[siddr] step {global_step} -> "
                      f"noisy={siddr_noisy:.2f} dB | unity={siddr_unity:.2f} dB | "
                      f"restored={si_sdr_db(yhat, clean).mean().item():.2f} dB | "
                      f"best noisy={best_noisy:.2f}@{k_n}, best restored={best_yhat:.2f}@{k_r}")
                comp = {
                    "mr": float(mr.detach()),
                    "l1": float(l1.detach()),
                    "si_ratio": float(si.detach()) if isinstance(si, torch.Tensor) else float(si),
                    "si_db": float(si_db_val),
                    "mel": float(mel.detach()) if isinstance(mel, torch.Tensor) else float(mel),
                    "hi": float(hi.detach()) if isinstance(hi, torch.Tensor) else float(hi),
                }
                print(
                    f"[comp] step {global_step} -> "
                    + " ".join(f"{k}={v:.4f}" for k, v in comp.items())
                )
                with torch.no_grad():
                    # Always re-compute Xn / y_unity here from the current `noisy`
                    Xn_chk, _ = stft_pair(noisy, win, n_fft, hop)
                    y_unity = istft_from(Xn_chk, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)

                    noisy_mono = noisy.mean(dim=1) if noisy.dim() == 3 else noisy  # (B,T) exactly what stft_pair used

                    si_noisy_clean = si_sdr_db(noisy_mono, clean).mean().item()
                    si_unity_clean = si_sdr_db(y_unity, clean).mean().item()
                    si_unity_noisy = si_sdr_db(y_unity, noisy_mono).mean().item()

                    # optional: sanity on raw difference
                    max_err = (y_unity - noisy_mono).abs().max().item()
                    mse_err = torch.mean((y_unity - noisy_mono) ** 2).item()

                print(f"[unity-check] noisy_vs_clean={si_noisy_clean:.2f} dB | "
                      f"unity_vs_clean={si_unity_clean:.2f} dB | unity_vs_noisy={si_unity_noisy:.2f} dB | "
                      f"max|Δ|={max_err:.2e} MSE={mse_err:.2e}")


                def _sig(sig):  # tiny checksum: signaled mean and std
                    return float(sig.mean()), float(sig.std())

                print("[ck] noisy μ,σ:", _sig(noisy_mono), "| clean μ,σ:", _sig(clean),
                      "| unity μ,σ:", _sig(y_unity))

                print(f"[unity] step {global_step} -> noisy={si_sdr_db(noisy_m, clean_m).mean().item():.2f} dB | "
                      f"unity={si_sdr_db(istft_from(Xn, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop), clean_m).mean().item():.2f} dB | "
                      f"restored={si_db_val:.2f} dB")

            used += 1
            tot += s
            last_loss = s

            # update bar (throttled postfix to avoid overhead)
            train_bar.update(1)
            if step_in_epoch % 25 == 0:
                train_bar.set_postfix(
                    loss=f"{s:.4f}", used=used, skip=skipped, lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

            if live and (used % 10 == 0):
                live.add_train(global_step, float(s))

            if global_step == 50:  # run once
                with torch.no_grad():
                    Xn, _ = stft_pair(noisy, win, n_fft, hop)
                    y_unity = istft_from(Xn, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)

                    noisy_mono = noisy.mean(dim=1) if noisy.dim() == 3 else noisy
                    si_unity_vs_noisy = si_sdr_db(y_unity, noisy_mono).mean().item()
                    max_err = (y_unity - noisy_mono).abs().max().item()
                    mse_err = torch.mean((y_unity - noisy_mono) ** 2).item()
                print(f"[unity-vs-noisy] SI={si_unity_vs_noisy:.2f} dB  max|Δ|={max_err:.3e}  MSE={mse_err:.3e}")

                y_id = istft_from(Xn, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)
                print("[unity-probe]",
                      "SI(noisy,clean)=", si_sdr_db(noisy, clean).mean().item(),
                      "SI(y_id,clean)=", si_sdr_db(y_id, clean).mean().item())
                with torch.no_grad():
                    # fp32, autocast OFF
                    with torch.autocast("cuda", dtype=torch.float32, enabled=False):
                        X_fp32 = torch.stft(noisy.mean(1).float(), n_fft=n_fft, hop_length=hop, window=win, center=True,
                                            return_complex=True)
                        y_fp32 = torch.istft(X_fp32, n_fft=n_fft, hop_length=hop, window=win, center=True,
                                             length=noisy.shape[-1])
                        si_fp32 = si_sdr_db(y_fp32, clean).mean().item()

                    # bf16, autocast ON
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        X_bf16 = torch.stft(noisy.mean(1), n_fft=n_fft, hop_length=hop, window=win, center=True,
                                            return_complex=True)
                        y_bf16 = torch.istft(X_bf16, n_fft=n_fft, hop_length=hop, window=win, center=True,
                                             length=noisy.shape[-1])
                        si_bf16 = si_sdr_db(y_bf16, clean).mean().item()

                print(f"[stft/istft] fp32={si_fp32:.2f} dB | bf16(auto)={si_bf16:.2f} dB")



            # next batch
            if use_prefetch:
                batch = prefetch.next()
            else:
                batch = next(it, None)

        train_bar.close()
        avg_tr = tot / max(1, used)

        # ---------- validation (robust) ----------
        net.eval()
        tot_v = 0.0
        used_v = 0
        skipped_v = 0
        clip_thr = float(cfg["loss"].get("val_loss_clip", 25.0))
        trim_frac = float(cfg["loss"].get("val_trim_frac", 0.02))
        vals = []

        # swap in EMA weights (if any), then restore
        ema_applied = False
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
            ema.apply_to(net)
            ema_applied = True
        try:
            with torch.no_grad(), torch.autocast(
                device_type=("cuda" if device.startswith("cuda") else "cpu"),
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                val_bar = tqdm(va_ld, desc=f"valid {epoch}", leave=False, dynamic_ncols=True)
                for noisy, clean, _, _ in val_bar:
                    noisy = noisy.to(device, non_blocking=True)
                    clean = clean.to(device, non_blocking=True)

                    # ---- 1) STFT in float32 OUTSIDE autocast ----
                    # ---- 1) STFT in float32 OUTSIDE autocast ----
                    # A) STFT (autocast disabled internally by stft_pair) → float32 complex
                    Xn, Xn_ri = stft_pair(noisy, win, n_fft=n_fft, hop=hop)
                    if channels_last_en:
                        Xn_ri = Xn_ri.contiguous(memory_format=torch.channels_last)

                    # B) net forward under autocast
                    with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=amp_dtype,
                                        enabled=amp_enabled):
                        Xn_ri_amp = Xn_ri.to(amp_dtype) if amp_enabled else Xn_ri
                        M = net(Xn_ri_amp)

                    # C) head mapping & effective-mask clamp in float32
                    Mr, Mi = M[:, 0].float(), M[:, 1].float()
                    Xr, Xi = Xn.real.float(), Xn.imag.float()

                    mask_variant = str(cfg["model"].get("mask_variant", "mag_sigm1")).lower()

                    if mask_variant == "mag_sigm1":
                        Mag = 1.0 + (torch.sigmoid(Mr) - 0.5)  # ~[0.5, 1.5]
                        R, I = Mag, torch.zeros_like(Mag)
                    elif mask_variant == "mag":
                        Mag = torch.sqrt(Mr ** 2 + Mi ** 2 + 1e-8)
                        R, I = Mag, torch.zeros_like(Mag)
                    elif mask_variant == "plain":
                        R, I = Mr, Mi
                    elif mask_variant == "delta1":
                        R, I = 1.0 + Mr, Mi
                    elif mask_variant == "mag_delta1":
                        Mag = 1.0 + torch.sqrt(Mr ** 2 + Mi ** 2 + 1e-8)
                        R, I = Mag, torch.zeros_like(Mag)
                    else:
                        raise ValueError(f"Unknown model.mask_variant={mask_variant!r}")

                    mlim = float(cfg["loss"].get("mask_limit", 0.0))
                    if mlim > 0:
                        mag_eff = torch.sqrt(R ** 2 + I ** 2 + 1e-8)
                        scale = torch.clamp(mlim / mag_eff, max=1.0)
                        R, I = R * scale, I * scale

                    Xhat = torch.complex(R * Xr - I * Xi, R * Xi + I * Xr)

                    # D) ISTFT (autocast disabled internally by istft_from)
                    yhat = istft_from(Xhat, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)

                    clean_m = _as_batch_mono(clean.float())  # (B,T)
                    noisy_m = _as_batch_mono(noisy.float())  # (B,T)

                    # === DEBUG AUDIO SAVE (first item of batch) ===
                    if (global_step % 100) == 0 or global_step == 0:
                        out_root = "debug_audio/train"
                        _ensure_dir(out_root)

                        # grab first item (mono) for simplicity
                        n0 = noisy[0]  # (C,T) or (T)
                        c0 = clean[0]
                        y0 = yhat[0]

                        # optional: unity recon to verify transform
                        Xn_chk, _ = stft_pair(noisy, win, n_fft, hop)
                        y_unity = istft_from(Xn_chk, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)[0]

                        # make a friendly stem: epoch_step_id
                        if isinstance(meta, dict) and "id" in meta and len(meta["id"]) > 0:
                            stem = f"e{epoch:03d}_s{global_step}_id"
                            # meta['id'] is probably a list; take first
                            try:
                                id0 = str(meta["id"][0]).split("/")[-1].split("\\")[-1]
                                stem += f"_{id0[:40]}"
                            except Exception:
                                pass
                        else:
                            stem = f"e{epoch:03d}_s{global_step}"

                        sr = int(cfg["data"]["sr"])

                        save_wav(os.path.join(out_root, f"{stem}_noisy.wav"), n0, sr)
                        save_wav(os.path.join(out_root, f"{stem}_clean.wav"), c0, sr)
                        save_wav(os.path.join(out_root, f"{stem}_pred.wav"), y0, sr)
                        save_wav(os.path.join(out_root, f"{stem}_unity.wav"), y_unity, sr)

                    # losses
                    lambda_phase = float(cfg["loss"].get("lambda_phase", 0.0))
                    phase = stft_phase_cosine(yhat, clean.float(), win, n_fft=n_fft,
                                              hop=hop) if lambda_phase > 0 else 0.0

                    mr_out = mrstft(yhat, clean.float())
                    mr = mr_out[0] if isinstance(mr_out, tuple) else mr_out
                    l1 = torch.mean(torch.abs(yhat - clean_m))
                    # SI-SDR (ratio) with epoch target + cap
                    sisdr_target_db = sisdr_target_for_epoch(cfg, epoch)
                    if float(cfg["loss"].get("lambda_sisdr", 0.0)) > 0:
                        si = si_sdr_ratio_loss(yhat, clean_m, min_db=sisdr_target_db) if lambda_sisdr > 0 else 0.0
                        si_db_val = si_sdr_db(yhat, clean_m).mean().item()
                        si_cap = float(cfg["loss"].get("sisdr_loss_cap", 2.0))
                        if isinstance(si, torch.Tensor):
                            si = torch.clamp(si, max=si_cap)
                    else:
                        si = 0.0

                    mel = (
                        log_mel_L1(
                            yhat,
                            clean.float(),
                            win,
                            sr=cfg["data"]["sr"],
                            n_mels=int(cfg["loss"].get("mel_bands", 64)),
                            n_fft=n_fft,
                            hop=hop,
                        )
                        if float(cfg["loss"].get("lambda_mel", 0.0)) > 0
                        else 0.0
                    )
                    hi = (
                        highband_mag_L1(
                            yhat,
                            clean.float(),
                            win,
                            sr=cfg["data"]["sr"],
                            cutoff_khz=float(cfg["loss"].get("hi_emph_khz", 8.0)),
                            n_fft=n_fft,
                            hop=hop,
                        )
                        if float(cfg["loss"].get("lambda_hi", 0.0)) > 0
                        else 0.0
                    )

                    loss_v = (
                            mr
                            + 0.5 * l1
                            + float(cfg["loss"].get("lambda_sisdr", 0.0))
                            * (si if isinstance(si, torch.Tensor) else torch.tensor(si, device=device))
                            + float(cfg["loss"].get("lambda_mel", 0.0))
                            * (mel if isinstance(mel, torch.Tensor) else torch.tensor(mel, device=device))
                            + float(cfg["loss"].get("lambda_hi", 0.0))
                            * (hi if isinstance(hi, torch.Tensor) else torch.tensor(hi, device=device))
                            + lambda_phase
                            * (phase if isinstance(phase, torch.Tensor) else torch.tensor(phase, device=device))
                    )

                    # ---- energy regularizer (keep output RMS near input RMS) ----
                    lambda_energy = float(cfg["loss"].get("lambda_energy", 0.0))
                    if lambda_energy > 0:
                        y_std = yhat.float().std(dim=-1).mean()
                        x_std = noisy_m.float().std(dim=-1).mean()
                        loss_v = loss_v + lambda_energy * ((y_std / (x_std + 1e-8)) - 1.0).pow(2)

                    lambda_idmask = float(cfg["loss"].get("lambda_idmask", 0.0))
                    if lambda_idmask > 0:
                        if mask_variant == "mag_sigm1":
                            reg = (R - 1.0).pow(2).mean()  # R is Mag here
                        elif mask_variant in ("mag", "mag_delta1"):
                            reg = (R - (1.0 if mask_variant == "mag" else 1.0)).pow(2).mean()
                        elif mask_variant == "plain":
                            reg = (Mr - 1.0).pow(2).mean() + (Mi - 0.0).pow(2).mean()
                        elif mask_variant == "delta1":
                            reg = (Mr - 0.0).pow(2).mean() + (Mi - 0.0).pow(2).mean()
                        else:
                            reg = torch.zeros((), device=yhat.device)
                        loss_v = loss_v + lambda_idmask * reg

                    s = float(loss_v.detach().item())
                    if (not math.isfinite(s)) or (clip_thr > 0 and s > clip_thr):
                        skipped_v += 1
                        continue
                    tot_v += s
                    used_v += 1
                    vals.append(s)
                    if used_v % 8 == 0:
                        val_bar.set_postfix(loss=f"{s:.4f}")
        finally:
            if ema_applied:
                net.load_state_dict(backup)

        avg_v = tot_v / max(1, used_v)
        if vals and trim_frac > 0:
            k = int(len(vals) * trim_frac)
            sv = sorted(vals)
            tv = sv[k: len(sv) - k] if len(sv) - 2 * k > 0 else sv
            avg_v_robust = sum(tv) / max(1, len(tv))
        else:
            avg_v_robust = avg_v

        if live:
            live.add_val(epoch, float(avg_v_robust))

        # ------ hard-mining weights update ------
        if use_hard and tr_sampler is not None and epoch >= start_hm_epoch and miner is not None:
            new_w = miner.make_weights(tr_ds)
            tr_sampler.set_weights(new_w)

        # ------ save checkpoint ------
        state = {
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        if ema:
            state["ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        out = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, out)

        # brief epoch summary (single line)
        print(
            f"[epoch {epoch:03d}] train {avg_tr:.4f} | val {avg_v_robust:.4f} | used {used}/{len(tr_ld)} "
            f"| skipped {skipped} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

    print("done.")


if __name__ == "__main__":
    main()
