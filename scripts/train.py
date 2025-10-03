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

    def freeze_batchnorm(m):
        import torch.nn as nn
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm3d):
                mod.eval()
                mod.track_running_stats = False

    # For overfit sanity only:
    if bool(cfg.get("debug", {}).get("freeze_bn", True)):
        freeze_batchnorm(model)
        print("[model] BatchNorm frozen for overfit sanity")
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
    # after optimizer creation
    constant_lr = bool(cfg.get("debug", {}).get("constant_lr", True))  # default True for overfit sanity
    if constant_lr:
        sched = None
        print("[optim] constant LR (scheduler disabled)")
    else:
        sched = _build_warmup_cosine(optim, total_steps, cfg["optim"]) if cfg["optim"].get("use_scheduler", True) else None



    # -----------------------
    # Callbacks
    # -----------------------
    cbs = []

    # 0) Procedural noise first — mutates 'noisy' before anyone audits/saves it
    try:
        from soundrestorer.callbacks.proc_noise_augment import ProcNoiseAugmentCallback
        from soundrestorer.callbacks.ensure_min_snr import EnsureMinSNRCallback
    except Exception:
        ProcNoiseAugmentCallback = None
        EnsureMinSNRCallback = None

    try:
        from soundrestorer.callbacks.save_preds import SavePredsEveryNStepsCallback
    except Exception:
        SavePredsEveryNStepsCallback = None

    cb_cfg = cfg.get("callbacks", {})

    if ProcNoiseAugmentCallback is not None and cb_cfg.get("proc_noise", {}).get("enable", False):
        # support both flattened and nested ("args") styles
        _pn = cb_cfg["proc_noise"]
        pn_args = dict(_pn.get("args", {})) if isinstance(_pn.get("args"), dict) else dict(_pn)

        # if not explicitly given here, fall back to data.out_peak
        out_peak = pn_args.get("out_peak", cfg.get("data", {}).get("out_peak", 0.98))
        # allow YAML null -> Python None to mean "no re-peak"
        out_peak = None if (out_peak is None or str(out_peak).lower() == "none") else float(out_peak)

        cbs.append(ProcNoiseAugmentCallback(
            sr=int(cfg["data"]["sr"]),
            prob=float(pn_args.get("prob", 0.5)),
            snr_min=float(pn_args.get("snr_min", 0.0)),
            snr_max=float(pn_args.get("snr_max", 20.0)),
            out_peak=out_peak,
            train_only=bool(pn_args.get("train_only", True)),
            noise_cfg=pn_args.get("noise_cfg", {}),
            track_stats=bool(pn_args.get("track_stats", True)),
            fixed_seed=pn_args.get("fixed_seed", None),
            fixed_per_epoch=bool(pn_args.get("fixed_per_epoch", False)),
            require_replace=bool(pn_args.get("require_replace", False)),
        ))
        print("[callbacks] ProcNoiseAugmentCallback enabled",
              f"(prob={pn_args.get('prob')}, snr=[{pn_args.get('snr_min')},{pn_args.get('snr_max')}], "
              f"fixed_seed={pn_args.get('fixed_seed')}, fixed_per_epoch={pn_args.get('fixed_per_epoch')}, "
              f"out_peak={out_peak})")

    # (optional) rescue items that are still too clean after dataset/proc:
    if EnsureMinSNRCallback is not None and cb_cfg.get("ensure_min_snr", {}).get("enable", False):
        em = cb_cfg["ensure_min_snr"]
        cbs.append(EnsureMinSNRCallback(
            sr=int(cfg["data"]["sr"]),
            min_snr_db=float(em.get("min_snr_db", 25.0)),
            snr_min=float(em.get("snr_min", 4.0)),
            snr_max=float(em.get("snr_max", 20.0)),
            out_peak=float(cfg["data"].get("out_peak", 0.98)),
            train_only=True,
        ))

    # 1) Data audit (now sees post-augment 'noisy')
    from soundrestorer.callbacks import DataAuditCallback, AudioDebugCallback
    if cb_cfg.get("data_audit", {}).get("enable", True):
        p = cb_cfg.get("data_audit", {})
        cbs.append(DataAuditCallback(
            max_items=int(p.get("max_items", 12)),
            take_from_batches=int(p.get("take_from_batches", 6)),
            subdir=str(p.get("subdir", "logs/data_audit")),
            sr=int(cfg["data"]["sr"]),
            silence_threshold=float(p.get("silence_threshold", 0.95)),
            write_wavs=bool(p.get("write_wavs", True)),
            write_csv=bool(p.get("write_csv", True)),
        ))

    # 2) Audio debug (validation-time triads)
    if cb_cfg.get("audio_debug", {}).get("enable", True):
        p = cb_cfg.get("audio_debug", {})
        cbs.append(AudioDebugCallback(
            per_epoch=int(p.get("per_epoch", 3)),
            scan_batches=int(p.get("scan_batches", 32)),
            seed=int(p.get("seed", 1234)),
            prefer_clean_resid=bool(p.get("prefer_clean_resid", True)),
            subdir=str(p.get("subdir", "logs/audio_debug")),
            sr=int(cfg["data"]["sr"]),
            print_metrics=True,
        ))

    sp = cb_cfg.get("save_preds", {})
    if SavePredsEveryNStepsCallback and sp.get("enable", False):
        args = sp.get("args", {})
        cbs.append(SavePredsEveryNStepsCallback(
            out_subdir=str(args.get("out_subdir", "logs/pred_dumps")),
            every_steps=int(args.get("every_steps", 100)),
            per_batch=int(args.get("per_batch", 1)),
            max_total=int(args.get("max_total", 200)),
            save_triads=bool(args.get("save_triads", False)),
            resid_mode=str(args.get("resid_mode", "noise")),
        ))
        print("[callbacks] SavePredsEveryNStepsCallback enabled")


    # -----------------------
    # Trainer
    # -----------------------
    amp = _infer_amp(cfg)
    trainer = Trainer(
        model=model,
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
        print_every=int(cfg.get("debug", {}).get("pbar_every", 50))
    )

    # Optional resume
    if args.get("resume", None):
        try:
            trainer.resume(Path(args.resume))
        except Exception as e:
            print(f"[resume][WARN] {e}")

    # banner (mirrors your style a bit)
    _print_training_banner(run_dir, cfg)

    # Eval-only branch
    if args.get("eval_only", False):
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
        overfit_steps=int(cfg.get("debug", {}).get("overfit_steps", 0)),
    )

if __name__ == "__main__":
    main()
