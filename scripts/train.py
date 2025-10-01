# -*- coding: utf-8 -*-
# scripts/train.py
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

# --- project imports (aligned with the rest of your code) ---
from soundrestorer.core.trainer import Trainer
from soundrestorer.callbacks import AudioDebugCallback, DataAuditCallback

# factories (assumed present from prior steps)
from soundrestorer.models.factory import create_model
from soundrestorer.tasks.factory import create_task
from soundrestorer.losses.composed import build_losses
from soundrestorer.data.builder import build_loaders  # returns (train_loader, val_loader, info)

# ADD (safe import; ignore if missing)
try:
    from soundrestorer.core.callbacks import ProcNoiseAugmentCallback
except Exception:
    ProcNoiseAugmentCallback = None

# -----------------------
# Helpers
# -----------------------

REQ_TOP_KEYS = ["data", "model", "task", "losses", "optim", "train", "runtime"]

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQ_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"[config error] Missing top-level keys: {missing}. "
                         f"Your YAML must define at least: {', '.join(REQ_TOP_KEYS)}")
    return cfg

def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _make_run_dir(runs_root: Path, run_name: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_root / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def _print_training_banner(run_dir: Path, cfg: Dict[str, Any]):
    print(f"Run dir: {run_dir}")
    mv = str(cfg["task"]["args"].get("mask_variant", "plain")).lower()
    print(f"[init] head bias set for mask_variant={mv}: "
          f"bias=[{cfg['task']['args'].get('head_bias_re', 0.0):+.4f}, "
          f"{cfg['task']['args'].get('head_bias_im', 0.0):+.4f}]")

def _build_optimizer(model: nn.Module, cfg_optim: Dict[str, Any]) -> Optimizer:
    lr = float(cfg_optim.get("lr", 3e-4))
    wd = float(cfg_optim.get("wd", 0.01))
    betas = cfg_optim.get("betas", (0.9, 0.999))
    eps = float(cfg_optim.get("eps", 1e-8))
    fused = bool(cfg_optim.get("fused", True))
    print(f"[optim] AdamW fused={fused}")
    return AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps, fused=fused)

def _build_warmup_cosine(optim: Optimizer, total_steps: int, cfg_optim: Dict[str, Any]) -> LambdaLR:
    warmup_steps = int(cfg_optim.get("warmup_steps", max(1, int(0.02 * total_steps))))
    min_factor = float(cfg_optim.get("lr_min_factor", 0.1))  # final LR multiplier
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine from 1 -> min_factor
        return min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * t))
    import math
    return LambdaLR(optim, lr_lambda)

def _infer_amp(cfg: Dict[str, Any]) -> str:
    amp = str(cfg.get("runtime", {}).get("amp", "bfloat16")).lower()
    # Normalize
    if amp in ("bf16", "bfloat16"): return "bfloat16"
    if amp in ("fp16", "float16", "half"): return "float16"
    return "off" if amp in ("off", "false", "0", "none") else amp

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser("SoundRestorer - train")
    ap.add_argument("--config", type=str, required=True, help="YAML config")
    ap.add_argument("--resume", type=str, default=None, help="checkpoint dir or file")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))

    # seed + device
    seed = int(cfg.get("seed", 1234) if args.seed is None else args.seed)
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run dir
    runs_root = Path(cfg.get("paths", {}).get("runs_root", "runs"))
    run_name = cfg.get("paths", {}).get("run_name", "train")
    run_dir = _make_run_dir(runs_root, run_name)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    # persist merged config for reference
    with open(run_dir / "configs" / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # -----------------------
    # Build data
    # -----------------------
    tr_loader, va_loader, data_info = build_loaders(cfg)
    data_sr = int(data_info.get("sr", cfg["data"].get("sr", 48000)))

    # -----------------------
    # Build model + task
    # -----------------------
    model = create_model(cfg["model"]["name"], **(cfg["model"].get("args") or {}))
    # Create task module that wraps the model and implements forward(batch)->outputs
    task = create_task(cfg["task"]["name"], model=model, **(cfg["task"].get("args") or {}))

    # -----------------------
    # Losses
    # -----------------------
    loss_fn = build_losses(cfg)

    # -----------------------
    # Optimizer + Scheduler
    # -----------------------
    optim = _build_optimizer(task, cfg["optim"])
    total_steps = (cfg["train"].get("epochs", 1)) * (cfg["debug"].get("overfit_steps", 0) or len(tr_loader))
    sched = _build_warmup_cosine(optim, total_steps, cfg["optim"]) if cfg["optim"].get("use_scheduler", True) else None

    # -----------------------
    # Callbacks
    # -----------------------
    cbs = []
    # wire defaults if enabled
    cb_cfg = cfg.get("callbacks", {})
    if cb_cfg.get("audio_debug", {}).get("enable", True):
        p = cb_cfg.get("audio_debug", {})
        cbs.append(AudioDebugCallback(
            per_epoch=int(p.get("per_epoch", 3)),
            scan_batches=int(p.get("scan_batches", 32)),
            seed=int(p.get("seed", 1234)),
            prefer_clean_resid=bool(p.get("prefer_clean_resid", True)),
            subdir=str(p.get("subdir", "logs/audio_debug")),
            sr=data_sr,
            print_metrics=True,
        ))
    if cb_cfg.get("data_audit", {}).get("enable", True):
        p = cb_cfg.get("data_audit", {})
        cbs.append(DataAuditCallback(
            max_items=int(p.get("max_items", 12)),
            take_from_batches=int(p.get("take_from_batches", 6)),
            subdir=str(p.get("subdir", "logs/data_audit")),
            sr=data_sr,
            silence_threshold=float(p.get("silence_threshold", 0.95)),
            write_wavs=bool(p.get("write_wavs", True)),
            write_csv=bool(p.get("write_csv", True)),
        ))

    # -----------------------
    # Trainer
    # -----------------------
    amp = _infer_amp(cfg)
    trainer = Trainer(
        task=task,
        loss_fn=loss_fn,
        optimizer=optim,
        scheduler=sched,
        run_dir=run_dir,
        device=device,
        amp=amp,
        grad_accum=int(cfg["optim"].get("grad_accum", 1)),
        grad_clip=float(cfg["optim"].get("grad_clip", cfg["optim"].get("grad_clip_norm", 0.0))),
        channels_last=bool(cfg["runtime"].get("channels_last", True)),
        compile=bool(cfg["runtime"].get("compile", False)),
        ema_beta=float(cfg["train"].get("ema", 0.0)),
        callbacks=cbs,
        print_every=int(cfg.get("debug", {}).get("pbar_every", 50)),
    )

    # Optional resume
    if args.resume:
        try:
            trainer.resume(Path(args.resume))
        except Exception as e:
            print(f"[resume][WARN] {e}")

    # banner (mirrors your style a bit)
    _print_training_banner(run_dir, cfg)

    # Eval-only branch
    if args.eval_only:
        if va_loader is None:
            print("[eval-only] No validation loader configured.")
            return
        val_loss, n_items = trainer._run_val_epoch(va_loader, epoch=trainer.state.epoch)
        print(f"[eval-only] val {val_loss:.4f} | used {n_items}")
        return

    # Fit
    trainer.fit(
        tr_loader,
        va_loader,
        epochs=int(cfg["train"]["epochs"]),
        save_every=int(cfg["train"].get("save_every", 1)),
        overfit_steps=int(cfg.get("debug", {}).get("overfit_steps", 0)),
    )

if __name__ == "__main__":
    main()
