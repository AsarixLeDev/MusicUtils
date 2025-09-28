#!/usr/bin/env python3
from __future__ import annotations
import argparse, torch, os

from soundrestorer.core.config import load_yaml, apply_overrides, prepare_run_dirs
from soundrestorer.core.registry import MODELS
from soundrestorer.core.trainer import Trainer, WarmupCosine
from soundrestorer.core.callbacks import (
    EMACallback, CheckpointCallback, ConsoleLogger, HardMiningCallback,
    ProcNoiseAugmentCallback, CurriculumCallback, AudioDebugCallback,
    BestCheckpointCallback, EarlyStoppingCallback, EpochSeedCallback
)
from soundrestorer.core.plugins import autoload_packages
from soundrestorer.losses.composed import ComposedLoss
from soundrestorer.data.builder import build_denoise_loader
from soundrestorer.core.loader import default_loader_args

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except: pass

def parse_args():
    ap = argparse.ArgumentParser("Generic Trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    return ap.parse_args()

def main():
    args = parse_args()
    cfg0 = load_yaml(args.config)
    cfg  = apply_overrides(cfg0, args.set)
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # autoload modules to fill registries
    autoload_packages([
        "soundrestorer.models",
        "soundrestorer.tasks",
        "soundrestorer.losses",
    ])

    paths = prepare_run_dirs(cfg)
    print(f"Run dir: {paths['root']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg["runtime"].get("amp", "bfloat16")
    ch_last = bool(cfg["runtime"].get("channels_last", True))
    compile_ok = bool(cfg["runtime"].get("compile", False))

    # ---- DATA ----
    tr_ds, tr_ld, tr_sampler = build_denoise_loader(cfg["data"]["train_manifest"], cfg["data"], train=True)
    va_ds, va_ld, _         = build_denoise_loader(cfg["data"]["val_manifest"],   cfg["data"], train=False)
    tr_args = default_loader_args(num_items=len(tr_ds), batch_size=cfg["data"]["batch"],
                                  workers_cfg=cfg["data"].get("workers"))
    va_args = default_loader_args(num_items=len(va_ds), batch_size=cfg["data"]["batch"],
                                  workers_cfg=cfg["data"].get("workers"))
    # validation: no shuffle, no drop_last
    va_args.update(shuffle=False, drop_last=False)

    tr_ld = torch.utils.data.DataLoader(tr_ds, **tr_args)
    va_ld = torch.utils.data.DataLoader(va_ds, **va_args)

    # ---- MODEL ----
    model = MODELS.build(cfg["model"]["name"], **cfg["model"].get("args", {})).float().to(device)

    from soundrestorer.models.init_utils import init_head_for_mask
    task_mask = cfg.get("task", {}).get("args", {}).get("mask_variant", "plain")
    init_head_for_mask(model, mask_variant=task_mask)

    # ---- TASK ----
    from soundrestorer.core.registry import TASKS
    task = TASKS.build(cfg["task"]["name"], **cfg["task"].get("args", {}))

    # ---- LOSSES ----
    loss_fn = ComposedLoss(cfg["losses"]["items"])

    # ---- OPT / SCHED ----
    opt_cfg = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(opt_cfg["lr"]),
                            betas=(0.9, 0.95),
                            eps=1e-8,
                            weight_decay=float(opt_cfg.get("wd", 0.0)))
    total_steps = int(cfg["train"]["epochs"]) * max(1, len(tr_ld))
    min_factor = float(cfg["optim"].get("lr_min_factor", 0.3))
    sched = WarmupCosine(opt, total_steps=total_steps,
                         warmup=int(opt_cfg.get("warmup_steps", 300)),
                         min_factor=min_factor)

    # ---- CALLBACKS ----
    cbs = [ConsoleLogger()]
    if float(cfg["train"].get("ema", 0.0)) > 0:
        cbs.append(EMACallback(model, decay=float(cfg["train"]["ema"])))
    if cfg["data"].get("hard_mining", {}).get("enable", False):
        hm = cfg.get("hard_mining", {})
        cbs.append(HardMiningCallback(
            dataset=tr_ds, sampler=tr_sampler,
            start_epoch=hm.get("start_epoch", 3),
            ema=hm.get("ema", 0.9),
            top_frac=hm.get("top_frac", 0.3),
            boost=hm.get("boost", 4.0),
            baseline=hm.get("baseline", 1.0),
        ))
    # checkpoint every epoch
    cbs.append(CheckpointCallback(paths["checkpoints"], every=int(cfg["train"].get("save_every", 1))))

    # ---- Optional extras from cfg.callbacks ----
    cb_cfg = cfg.get("callbacks", {})

    # procedural noise (train-time batch augment)
    if cb_cfg.get("proc_noise", {}).get("enable", False):
        pn = cb_cfg["proc_noise"]
        cbs.append(ProcNoiseAugmentCallback(
            sr=int(cfg["data"]["sr"]),
            prob=float(pn.get("prob", 0.5)),
            snr_min=float(pn.get("snr_min", 0.0)),
            snr_max=float(pn.get("snr_max", 20.0)),
            out_peak=float(cfg["data"].get("out_peak", 0.98)),
            train_only=True,
            noise_cfg=pn.get("noise_cfg", {})
        ))

    # curriculum (dataset + loss + task)
    if cb_cfg.get("curriculum", {}).get("enable", False):
        cur = cb_cfg["curriculum"]
        cbs.append(CurriculumCallback(
            loss_fn=loss_fn,
            task=task,
            dataset=tr_ds,
            snr_stages=cur.get("snr_stages", []),
            sisdr=cur.get("sisdr", {}),
            mask_limit=cur.get("mask_limit", {}),
            mask_variant=cur.get("mask_variant", "plain"),
        ))

    # audio debug
    if cb_cfg.get("audio_debug", {}).get("enable", False):
        ad = cb_cfg["audio_debug"]
        cbs.append(AudioDebugCallback(
            out_dir=paths["logs"], val_dataset=va_ds, task=task, sr=int(cfg["data"]["sr"]),
            every=int(ad.get("every", 5)), num_items=int(ad.get("num_items", 3))
        ))

    # best-K checkpoints
    if cb_cfg.get("best_ckpt", {}).get("enable", False):
        bc = cb_cfg["best_ckpt"]
        cbs.append(BestCheckpointCallback(
            out_dir=paths["checkpoints"], k=int(bc.get("top_k", 3)),
            monitor=bc.get("monitor", "val_loss"), mode=bc.get("mode", "min")
        ))

    # early stopping
    if cb_cfg.get("early_stop", {}).get("enable", False):
        es = cb_cfg["early_stop"]
        cbs.append(EarlyStoppingCallback(
            patience=int(es.get("patience", 10)),
            min_delta=float(es.get("min_delta", 0.0)),
            monitor=es.get("monitor", "val_loss"),
            mode=es.get("mode", "min")
        ))

    # epoch reseed (varied synthetic noise each epoch)
    if cb_cfg.get("epoch_seed", {}).get("enable", True):
        cbs.append(EpochSeedCallback([tr_ds, va_ds]))

    # ---- TRAINER ----
    trainer = Trainer(
        model=model, task=task, loss_fn=loss_fn, optimizer=opt, scheduler=sched,
        device=device, amp=amp, grad_accum=int(opt_cfg.get("grad_accum", 1)),
        grad_clip=float(opt_cfg.get("grad_clip", 0.0)), channels_last=ch_last,
        compile_model=compile_ok, callbacks=cbs
    )
    # make sure runtime + guard knobs are visible to trainer.fit()
    trainer.runtime = cfg.get("runtime", {})
    trainer.guard_cfg = cfg.get("guard", {})
    trainer.debug_cfg = cfg.get("debug", {})  # <— NEW
    trainer.train_loss_clip = cfg.get("loss", {}).get("train_loss_clip", 0.0)
    trainer.train_trim_frac = cfg.get("loss", {}).get("train_trim_frac", 0.10)

    # ---- GO ----
    trainer.fit(tr_ld, va_ld, epochs=int(cfg["train"]["epochs"]), ckpt_saver=None)

if __name__ == "__main__":
    main()
