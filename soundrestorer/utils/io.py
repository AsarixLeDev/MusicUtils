# -*- coding: utf-8 -*-
# soundrestorer/utils/io.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import torch

try:
    import torchaudio

    _HAS_TA = True
except Exception:
    _HAS_TA = False


def read_wav(path: str | Path, sr: Optional[int] = None, mono: Optional[bool] = None) -> torch.Tensor:
    """
    Returns waveform as [C, T] float32 in [-1, 1].
    Optionally resamples to `sr` and converts to mono.
    """
    if not _HAS_TA:
        raise RuntimeError("torchaudio is required for read_wav")
    wav, file_sr = torchaudio.load(str(path))  # [C,T], dtype float32/16-bit -> float32
    if (sr is not None) and (int(file_sr) != int(sr)):
        wav = torchaudio.functional.resample(wav, file_sr, sr)
        file_sr = sr
    if mono is True and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def write_wav(path: str | Path, wav: torch.Tensor, sr: int):
    """
    Save waveform tensor. Accepts [T], [C,T], [B,C,T] (B>1 saves first item).
    """
    if not _HAS_TA:
        raise RuntimeError("torchaudio is required for write_wav")
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True)
    x = wav
    if x.dim() == 3:
        x = x[0]  # take first item
    if x.dim() == 1:
        x = x.unsqueeze(0)
    torchaudio.save(str(p), x.to(torch.float32), sr)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]):
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]):
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
