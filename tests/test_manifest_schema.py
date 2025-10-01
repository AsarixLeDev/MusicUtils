# tests/test_manifest_schema.py
import json, os
from pathlib import Path
import torch

from soundrestorer.utils.io import write_wav, read_wav
from soundrestorer.utils.audio import to_mono

def _mk_wav(path: Path, sr=8000, seconds=1.0, freq=220.0):
    T = int(sr * seconds)
    t = torch.linspace(0, seconds, T)
    x = 0.1 * torch.sin(2 * torch.pi * freq * t)
    write_wav(path, x.unsqueeze(0), sr)

def test_manifest_schema_and_durations(tmp_path: Path):
    d = tmp_path / "a"; d.mkdir()
    sr = 8000
    clean = d / "c.wav"
    noisy = d / "n.wav"
    _mk_wav(clean, sr=sr, seconds=1.0, freq=220.0)
    _mk_wav(noisy, sr=sr, seconds=1.0, freq=330.0)

    man = d / "manifest.jsonl"
    rows = [
        {"id": "ex1", "split": "train", "clean": str(clean), "noisy": str(noisy), "sr": sr, "duration": 1.0},
        {"id": "ex2", "split": "val", "clean": str(clean), "noisy": str(noisy), "sr": sr, "duration": 1.0},
    ]
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Schema & quick checks
    with open(man, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert "id" in r and "split" in r and "clean" in r and "noisy" in r and "sr" in r
            assert r["split"] in {"train", "val", "test"}
            # Check duration via samples
            wav_c = read_wav(r["clean"], sr=sr, mono=False)
            assert wav_c.shape[-1] == sr  # ~1.0s
            wav_m = to_mono(wav_c)
            assert wav_m.dim() == 2 and wav_m.shape[0] == 1
