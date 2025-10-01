# tests/test_callbacks.py
import os
from pathlib import Path
import pytest
import torch

def test_audio_debug_and_data_audit(tmp_path: Path):
    adc = pytest.importorskip("soundrestorer.callbacks.audio_debug", reason="audio_debug callback missing")
    dac = pytest.importorskip("soundrestorer.callbacks.data_audit", reason="data_audit callback missing")
    from soundrestorer.callbacks.audio_debug import AudioDebugCallback
    from soundrestorer.callbacks.data_audit import DataAuditCallback

    # synthetic batch
    B, T = 1, 16000
    clean = torch.randn(B,1,T)*0.05
    noisy = clean + 0.2*torch.randn_like(clean)
    yhat  = noisy*0.8
    batch = {"clean": clean, "noisy": noisy, "sr": 16000}
    outputs = {"yhat": yhat}

    out1 = tmp_path / "dbg"
    out2 = tmp_path / "audit"

    c1 = AudioDebugCallback(root=out1, every_n_epochs=1, max_items=2)
    c1.on_validation_batch_end(outputs, batch, epoch=1, batch_idx=0)

    c2 = DataAuditCallback(root=out2, every_n_epochs=1, max_items=2)
    c2.on_validation_batch_end(outputs, batch, epoch=1, batch_idx=0)

    # files present?
    triad_files = list(out1.rglob("*_yhat.wav")) + list(out1.rglob("*_noisy.wav")) + list(out1.rglob("*_clean.wav"))
    assert len(triad_files) >= 3
    assert (out2 / "ep001" / "audit.csv").exists() or (out2 / "audit.csv").exists()

def test_perceptual_eval_callback_if_present(tmp_path: Path):
    try:
        from soundrestorer.callbacks.perceptual_eval import PerceptualEvalCallback
    except Exception:
        pytest.skip("perceptual_eval callback not present")
    # Smoke
    c = PerceptualEvalCallback(root=tmp_path / "peval", every_n_epochs=1, max_items=1)
    B, T = 1, 16000
    clean = torch.randn(B,1,T)*0.05
    noisy = clean + 0.1*torch.randn_like(clean)
    yhat = noisy*0.9
    batch = {"clean": clean, "noisy": noisy, "sr": 16000}
    outputs = {"yhat": yhat}
    c.on_validation_batch_end(outputs, batch, epoch=1, batch_idx=0)
    assert (tmp_path / "peval").exists()
