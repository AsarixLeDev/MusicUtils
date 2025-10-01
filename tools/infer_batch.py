# -*- coding: utf-8 -*-
# tools/infer_batch.py
"""
Run a trained checkpoint on a folder or manifest and dump:
  - <name>_noisy.wav, <name>_yhat.wav (and optionally <name>_clean.wav if available)
  - metrics CSV (SI-SDR, LSD, residual energy %, composite score)

Assumptions:
- You pass a training YAML `--config` that defines the model + task the same way as train.py.
- We try to import builder functions compatible with your codebase.
  Edit the IMPORT BLOCK below if your factories live elsewhere.

If factories are not found, we fall back to a passthrough "identity" model so you can
still generate metrics for existing triads.

Examples:
  python tools/infer_batch.py --config configs/denoiser_flexible_v2.yaml \
      --checkpoint runs/....pt --in ./noisy_folder --out ./infer_out
  python tools/infer_batch.py --config configs/denoiser_flexible_v2.yaml \
      --checkpoint runs/....pt --manifest data/val.jsonl --out ./infer_out
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from soundrestorer.utils.io import read_wav, write_wav, write_csv, read_jsonl
from soundrestorer.utils.audio import ensure_3d, match_length, to_mono, normalize_peak
from soundrestorer.utils.metrics import si_sdr_db, mae
from soundrestorer.utils.signal import stft_complex

# -------------------- IMPORT BLOCK (edit if needed) --------------------

def _try_factories():
    builders = {}
    tried = []
    for path in [
        ("soundrestorer.core.factory", "build_model", "build_task"),
        ("soundrestorer.models.factory", "build_model", None),
        ("soundrestorer.tasks.factory", None, "build_task"),
    ]:
        mod, bm, bt = path
        try:
            m = __import__(mod, fromlist=["*"])
            if bm and hasattr(m, bm): builders["build_model"] = getattr(m, bm)
            if bt and hasattr(m, bt): builders["build_task"] = getattr(m, bt)
        except Exception as e:
            tried.append((mod, str(e)))
    return builders

def _identity_task_forward(wav: torch.Tensor, sr: int) -> torch.Tensor:
    return wav

# ----------------------------------------------------------------------

def _lsd_db(a: torch.Tensor, b: torch.Tensor, n_fft: int = 2048, hop: int = 512, eps: float = 1e-8) -> float:
    A = stft_complex(ensure_3d(a), n_fft=n_fft, hop_length=hop)
    B = stft_complex(ensure_3d(b), n_fft=n_fft, hop_length=hop)
    Am = (A.abs().clamp_min(eps)).log10().mul(20.0)
    Bm = (B.abs().clamp_min(eps)).log10().mul(20.0)
    d = (Am - Bm).pow(2).mean(dim=-2).clamp_min(0).sqrt().mean()
    return float(d.item())

def _percent_residual_energy(clean: torch.Tensor, noisy: torch.Tensor, yhat: torch.Tensor) -> Optional[float]:
    a3, b3 = match_length(ensure_3d(clean), ensure_3d(noisy))
    y3, _  = match_length(ensure_3d(yhat), a3)
    num = (y3 - a3).pow(2).mean().item()
    den = (b3 - a3).pow(2).mean().item()
    if den <= 1e-12: return None
    return float(100.0 * (num / den))

def _sigmoid(x: float) -> float: return 1.0 / (1.0 + math.exp(-x))

def _composite(si_impr_db: float, resid_pct: Optional[float], lsd_impr_db: float) -> float:
    si_term = _sigmoid(si_impr_db / 3.0)
    rp_term = 0.5 if resid_pct is None else max(0.0, 1.0 - resid_pct / 100.0)
    lsd_term = _sigmoid(lsd_impr_db / 3.0)
    return 100.0 * (0.45 * si_term + 0.35 * lsd_term + 0.20 * rp_term)

def _collect_inputs_from_folder(folder: Path) -> List[Dict]:
    wavs = sorted(folder.rglob("*.wav"))
    return [{"id": w.stem, "noisy": str(w)} for w in wavs]

def _collect_inputs_from_manifest(manifest: Path) -> List[Dict]:
    rows = read_jsonl(manifest)
    items = []
    for i, r in enumerate(rows):
        item = {}
        if "noisy" in r: item["noisy"] = r["noisy"]
        elif "mixture" in r: item["noisy"] = r["mixture"]
        else: continue
        item["id"] = r.get("id", f"item_{i:06d}")
        if "clean" in r: item["clean"] = r["clean"]
        items.append(item)
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--in", dest="indir", type=str, help="folder of WAVs (assumed 'noisy')")
    grp.add_argument("--manifest", type=str, help="JSONL manifest with 'noisy' (or 'mixture') and optional 'clean'")
    ap.add_argument("--out", type=str, required=True, help="output folder")
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Build model & task
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    factories = _try_factories()
    model = None; task = None

    if "build_model" in factories and "build_task" in factories:
        model = factories["build_model"](cfg["model"])
        task  = factories["build_task"](cfg["task"])
        sd = torch.load(args.checkpoint, map_location="cpu")
        # allow various checkpoint formats
        sd_model = sd.get("model", sd)
        missing = model.load_state_dict(sd_model, strict=False)
        if isinstance(missing, tuple):
            print("[ckpt] load result:", missing)
        model.to(args.device).eval()
    else:
        print("[infer] WARNING: factories not found; using identity passthrough.")
        def task(x, sr): return x  # noqa

    # Collect inputs
    if args.indir:
        items = _collect_inputs_from_folder(Path(args.indir))
    else:
        items = _collect_inputs_from_manifest(Path(args.manifest))
    if not items:
        raise SystemExit("No inputs found.")

    results = []
    for it in items:
        noisy = read_wav(it["noisy"], sr=args.sr, mono=args.mono)
        noisy = normalize_peak(noisy, 0.98)
        B = 1; wav = noisy.unsqueeze(0) if noisy.dim()==2 else noisy  # [C,T] -> [1,C,T]
        sr = args.sr  # could be None; used only for write_wav naming

        with torch.no_grad():
            if model is not None:
                # assume task.forward(model, batch) -> outputs["yhat"]
                try:
                    batch = {"noisy": wav.to(args.device), "sr": sr}
                    outputs = task.forward(model, batch) if hasattr(task, "forward") else task(model, batch)
                    yhat = outputs["yhat"].detach().cpu()
                except Exception as e:
                    print(f"[infer] task forward failed, fallback identity: {e}")
                    yhat = wav
            else:
                yhat = wav

        # Save WAVs
        noisy_out = outdir / f"{it['id']}_noisy.wav"
        yhat_out  = outdir / f"{it['id']}_yhat.wav"
        write_wav(noisy_out, noisy, sr or 48000)
        write_wav(yhat_out,  yhat,  sr or 48000)

        # Metrics if clean present
        row = {"id": it["id"], "noisy": it["noisy"], "yhat": str(yhat_out)}
        if "clean" in it:
            clean = read_wav(it["clean"], sr=args.sr, mono=args.mono)
            clean, noisy = match_length(clean, noisy)
            clean, yhat  = match_length(clean, yhat)
            si_nc = float(si_sdr_db(noisy, clean)[0].item())
            si_yc = float(si_sdr_db(yhat,  clean)[0].item())
            d_si  = si_yc - si_nc
            lsd_nc = _lsd_db(noisy, clean)
            lsd_yc = _lsd_db(yhat,  clean)
            resid_pct = _percent_residual_energy(clean, noisy, yhat)
            comp = _composite(d_si, resid_pct, max(0.0, lsd_nc - lsd_yc))

            row.update(dict(
                clean=it["clean"],
                si_noisy_clean_db=si_nc,
                si_yhat_clean_db=si_yc,
                delta_si_db=d_si,
                lsd_noisy_clean_db=lsd_nc,
                lsd_yhat_clean_db=lsd_yc,
                lsd_impr_db=(lsd_nc - lsd_yc),
                resid_pct=resid_pct if resid_pct is not None else float("nan"),
                mae_yhat_clean=float(mae(ensure_3d(yhat), ensure_3d(clean)).item()),
                composite_0_100=comp,
            ))
        results.append(row)
        print(f"[infer] {it['id']} -> saved {yhat_out.name}")

    # CSV
    csv_path = outdir / "infer_metrics.csv"
    write_csv(csv_path, results, fieldnames=list(results[0].keys()))
    print(f"Wrote {csv_path}")

if __name__ == "__main__":
    main()
