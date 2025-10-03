# -*- coding: utf-8 -*-
"""
MusicDenoiseDataset
- Reads a JSONL manifest with "clean" paths (mixture from MUSDB/Moises).
- Optionally reads one or more noise manifests with "noise" paths.
- At __getitem__, loads/crops audio, samples a noise, mixes at target SNR, returns dict:
    { "noisy": T x 1, "clean": T x 1, "sr": int, "meta": {...} }
Compatible with trainer that expects keys "noisy" and "clean".
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from ..utils.audio import to_mono
from ..utils.metrics import rms_db
from soundrestorer.utils.audio import ensure_3d
from soundrestorer.utils.metrics import rms_db as _rms_db

def _rms_db_float(x: torch.Tensor) -> float:
    """
    Robust RMS dB as a plain Python float for any shape [T]/[C,T]/[B,C,T].
    Averages across channels/items so the result is scalar.
    """
    x3 = ensure_3d(x)                 # -> [B,C,T]
    val = _rms_db(x3, dim=(-1,-2))    # dB per item [B]
    return float(val.mean().detach().cpu().item())


# ------------------- small helpers -------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.mean(x ** 2), min=eps))


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def _crop_or_pad(wav: torch.Tensor, target_len: int, rng: random.Random, fixed_crop_start: int) -> torch.Tensor:
    # wav: (1, T)
    T = wav.size(-1)
    if T == target_len:
        return wav
    if T > target_len:
        if fixed_crop_start is not None:
            start = fixed_crop_start
        else:
            start = rng.randint(0, max(0, T - target_len))
        return wav[..., start:start + target_len]
    # pad (loop or zero)
    reps = (target_len + T - 1) // T
    wav2 = wav.repeat(1, reps)[..., :target_len]
    return wav2


def _peak_scale_pair(noisy: torch.Tensor, clean: torch.Tensor, peak: float = 0.98) -> Tuple[torch.Tensor, torch.Tensor]:
    peak_now = max(noisy.abs().amax().item(), 1e-9)
    if peak_now <= peak:
        return noisy, clean
    scale = peak / peak_now
    return noisy * scale, clean * scale


# ------------------- dataset -------------------

@dataclass
class MusicDenoiseCfg:
    sr: int = 48000
    crop: float = 3.0
    mono: bool = True
    snr_min: float = 0.0
    snr_max: float = 20.0
    use_ext_noise_p: float = 0.6
    p_clean: float = 0.05
    out_peak: float = 0.98
    silence_rms_db: float = -60.0  # reject overly silent crops
    max_silence_retries: int = 6


class MusicDenoiseDataset(Dataset):
    def __init__(self,
                 manifest_clean: Path,
                 cfg_data: Dict[str, Any],
                 noise_manifests: Optional[List[Path]] = None,
                 train: bool = True):
        super().__init__()
        self._fixed_crop_start: int | None = None  # in samples; None = random

        self.clean_rows = _load_jsonl(Path(manifest_clean))
        self.train = train

        # parse cfg
        self.cfg = MusicDenoiseCfg(
            sr=int(cfg_data.get("sr", 48000)),
            crop=float(cfg_data.get("crop", 3.0)),
            mono=bool(cfg_data.get("mono", True)),
            snr_min=float(cfg_data.get("snr_min", 0.0)),
            snr_max=float(cfg_data.get("snr_max", 20.0)),
            use_ext_noise_p=float(cfg_data.get("use_ext_noise_p", 0.6)),
            p_clean=float(cfg_data.get("p_clean", 0.05)),
            out_peak=float(cfg_data.get("out_peak", 0.98)),
            silence_rms_db=float(cfg_data.get("silence_rms_db", -60.0)),
            max_silence_retries=int(cfg_data.get("max_silence_retries", 6)),
        )

        # optional noise rows
        self.noise_rows: List[Dict[str, Any]] = []
        nm = noise_manifests or cfg_data.get("noise_manifests", [])
        for m in nm:
            self.noise_rows.extend(_load_jsonl(Path(m)))
        # pre-create RNG
        self.rng = random.Random()

    def set_fixed_crop_sec(self, start_sec: float | int | None):
        if start_sec is None:
            self._fixed_crop_start = None
        else:
            start = float(start_sec)
            self._fixed_crop_start = max(0, int(round(start * self.cfg.sr)))

    def __len__(self) -> int:
        return len(self.clean_rows)

    # --------- core IO helpers ---------
    def _load_wave(self, path: str) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)  # (C, T), sr
        return wav, sr

    def _prepare_clean(self, row: Dict[str, Any], T_target: int) -> torch.Tensor:
        wav, sr = self._load_wave(row["clean"])
        if self.cfg.mono:
            wav = to_mono(wav)
        wav = _resample_if_needed(wav, sr, self.cfg.sr)
        # retry a few times to avoid silent segments if possible
        for _ in range(self.cfg.max_silence_retries):
            crop = _crop_or_pad(wav, T_target, self.rng, self._fixed_crop_start)
            if _rms_db_float(crop) > self.cfg.silence_rms_db:
                return crop
        return _crop_or_pad(wav, T_target, self.rng, self._fixed_crop_start)

    def _sample_noise_wave(self, T_target: int) -> Optional[torch.Tensor]:
        if not self.noise_rows:
            return None
        row = self.rng.choice(self.noise_rows)
        wav, sr = self._load_wave(row["noise"])
        if self.cfg.mono:
            wav = to_mono(wav)
        wav = _resample_if_needed(wav, sr, self.cfg.sr)
        return _crop_or_pad(wav, T_target, self.rng, self._fixed_crop_start)

    # --------- mixing ---------
    def _mix(self, clean: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Return noisy, snr_db, noise_rms_db."""
        T = clean.size(-1)
        use_noise = self.rng.random() < self.cfg.use_ext_noise_p and self.noise_rows
        if (not use_noise) or (self.rng.random() < self.cfg.p_clean):
            # no external noise (identity sample)
            return clean.clone(), float("inf"), float("-inf")

        noise = self._sample_noise_wave(T)
        if noise is None:
            return clean.clone(), float("inf"), float("-inf")

        # DC removal (mild)
        clean = clean - clean.mean()
        noise = noise - noise.mean()

        # choose SNR
        snr_db = self.rng.uniform(self.cfg.snr_min, self.cfg.snr_max)
        c_rms = _rms(clean)
        n_rms = _rms(noise)
        eps = 1e-8
        if n_rms.item() < eps:
            return clean.clone(), float("inf"), float("-inf")

        # scale noise to target SNR: snr = 20 log10(c_rms / (g * n_rms)) => g = c_rms / (n_rms * 10^(snr/20))
        g = (c_rms / (n_rms * (10.0 ** (snr_db / 20.0)))).item()
        noisy = clean + noise * g

        # peak protection (scale both to preserve SNR)
        noisy, clean = _peak_scale_pair(noisy, clean, peak=self.cfg.out_peak)
        return noisy, float(snr_db), rms_db(noise * g)

    # --------- Dataset API ---------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.clean_rows[idx % len(self.clean_rows)]
        T_target = int(round(self.cfg.crop * self.cfg.sr)) if self.cfg.crop and self.cfg.crop > 0 else None

        wav_clean, sr_src = self._load_wave(row["clean"])
        if self.cfg.mono:
            wav_clean = to_mono(wav_clean)
        wav_clean = _resample_if_needed(wav_clean, sr_src, self.cfg.sr)
        if T_target:
            # try not-silent crop
            for _ in range(self.cfg.max_silence_retries):
                crop = _crop_or_pad(wav_clean, T_target, self.rng, self._fixed_crop_start)
                if _rms_db_float(crop) > self.cfg.silence_rms_db:
                    wav_clean = crop
                    break
            else:
                wav_clean = _crop_or_pad(wav_clean, T_target, self.rng, self._fixed_crop_start)

        noisy, snr_db, noise_rms_db = self._mix(wav_clean)

        # return shape (Batched later): (T,) waveform—match trainer expectations
        return {
            "noisy": noisy.squeeze(0),
            "clean": wav_clean.squeeze(0),
            "sr": self.cfg.sr,
            "meta": {
                "idx": idx,
                "snr_db": snr_db,
                "noise_rms_db": noise_rms_db,
                "src": row.get("source", "music"),
                "path": row.get("clean"),
            }
        }


# ------------------- Loader builder -------------------

def build_music_denoise_loader(manifest: str,
                               data_cfg: Dict[str, Any],
                               train: bool = True):
    """
    Return (dataset, dataloader, sampler_like_None) matching your trainer’s expected tuple.
    """
    batch = int(data_cfg.get("batch", 8))
    workers = int(data_cfg.get("workers", 0))
    prefetch = int(data_cfg.get("prefetch_factor", 2))
    pin = bool(data_cfg.get("pin_memory", True))
    persistent = bool(data_cfg.get("persistent_workers", False))
    nm = data_cfg.get("noise_manifests", [])

    ds = MusicDenoiseDataset(manifest_clean=Path(manifest),
                             cfg_data=data_cfg,
                             noise_manifests=[Path(x) for x in nm],
                             train=train)
    ld = DataLoader(ds,
                    batch_size=batch,
                    shuffle=train,
                    num_workers=workers,
                    pin_memory=pin,
                    persistent_workers=persistent,
                    prefetch_factor=prefetch if workers > 0 else None,
                    drop_last=train)
    return ds, ld, None
