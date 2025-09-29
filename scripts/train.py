#!/usr/bin/env python3
from __future__ import annotations
import argparse, inspect, os, re, glob, math, torch

from soundrestorer.core.config import load_yaml, apply_overrides, prepare_run_dirs
from soundrestorer.core.registry import MODELS
from soundrestorer.core.trainer import Trainer, WarmupCosine
from soundrestorer.core.callbacks import (
    CheckpointCallback, ConsoleLogger, HardMiningCallback,
    ProcNoiseAugmentCallback, CurriculumCallback, AudioDebugCallback,
    BestCheckpointCallback, EarlyStoppingCallback, EpochSeedCallback,
    EmaEvalSwap, EmaUpdateCallback
)
from soundrestorer.core.plugins import autoload_packages
from soundrestorer.ema.ema import EMA
from soundrestorer.losses.composed import ComposedLoss
from soundrestorer.data.builder import build_denoise_loader
from soundrestorer.core.loader import default_loader_args

# --- backend prefs ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

try:
    import torch._inductor.config as _inductor_cfg
    _inductor_cfg.max_autotune_gemm = False
except Exception:
    pass


# --- compile helper: try inductor, fallback to aot_eager, else off ---
def maybe_compile_model(model, runtime_cfg):
    want_compile = bool(runtime_cfg.get("compile", False))
    backend = str(runtime_cfg.get("compile_backend", "inductor")).lower()
    mode    = str(runtime_cfg.get("compile_mode", "default")).lower()

    if not want_compile or not hasattr(torch, "compile"):
        return model, False, "off"

    if backend == "inductor":
        try:
            import triton  # noqa: F401
        except Exception:
            print("[compile] Triton not available; falling back to backend='aot_eager'")
            backend = "aot_eager"

    try:
        m = torch.compile(model, backend=backend, mode=mode)
        print(f"[compile] torch.compile active ({backend}, {mode})")
        return m, True, f"{backend}/{mode}"
    except Exception as e:
        print(f"[compile] disabled ({backend}, {mode}): {e}")
        return model, False, "off"


def parse_args():
    ap = argparse.ArgumentParser("Generic Trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", default=None,
                    help="Path to a checkpoint file or a directory containing checkpoints.")
    return ap.parse_args()


def _adamw_param_groups(model: torch.nn.Module, weight_decay: float):
    """Exclude bias/1D params from weight decay (better AdamW defaults)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _find_latest_ckpt(path: str) -> str:
    """Return a checkpoint path. If 'path' is a dir, pick the latest epoch_XXX.pt;
    if none found, pick the most recent .pt by mtime."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"--resume path not found: {path}")

    candidates = glob.glob(os.path.join(path, "*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt files in directory: {path}")

    def epoch_of(p: str) -> int:
        b = os.path.basename(p)
        m = re.search(r"epoch[_\-]?(\d+)\.pt", b) or re.search(r"ep(?:och)?[_\-]?(\d+)\.pt", b)
        return int(m.group(1)) if m else -1

    with_epoch = [p for p in candidates if epoch_of(p) >= 0]
    if with_epoch:
        best = max(with_epoch, key=lambda p: epoch_of(p))
        return best
    # fallback: newest by mtime
    return max(candidates, key=os.path.getmtime)


def _align_warmupcosine_after_resume(sched: WarmupCosine, global_step: int):
    """Align our WarmupCosine so that the next sched.step() continues smoothly.
    We keep current optimizer LR as 'lr_now'; set sched._base so that base*s(t) == lr_now,
    where s(t) is the scale at 'global_step'."""
    if not isinstance(sched, WarmupCosine):
        return
    t = int(max(0, global_step))
    # compute s(t) exactly like WarmupCosine
    if t <= sched.warm:
        s = t / max(1, sched.warm)
    else:
        prog = (t - sched.warm) / max(1, sched.tot - sched.warm)
        s = sched.minf + (1 - sched.minf) * 0.5 * (1 + math.cos(math.pi * prog))
    s = float(max(1e-12, s))
    # set internal step and recompute bases so that base*s == current lr
    sched._step = t
    new_base = []
    for pg in sched.opt.param_groups:
        lr_now = float(pg.get("lr", 0.0))
        new_base.append(lr_now / s)
    sched._base = new_base


def _load_resume(ckpt_path: str, model, opt, sched, device="cuda"):
    """Load model/optimizer/scheduler, coercing the checkpoint to fit the current model when possible.
    Returns (start_epoch, global_step)."""
    import re
    ckpt = torch.load(ckpt_path, map_location=device)
    sd_ckpt = ckpt["model"]
    sd_model = model.state_dict()

    # ---- helpers to normalize prefixes ----
    PREFIXES = ("_orig_mod.", "module.")

    def strip_prefixes(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for p in PREFIXES:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
            out[nk] = v
        return out

    def add_module_prefix_if_needed(d: dict[str, torch.Tensor], model_keys) -> dict[str, torch.Tensor]:
        # If ALL model keys start with 'module.' and NONE of d's keys do, add it.
        has_mod_model = all(k.startswith("module.") for k in model_keys)
        has_mod_ckpt  = all(k.startswith("module.") for k in d.keys())
        if has_mod_model and not has_mod_ckpt:
            return { "module."+k: v for k, v in d.items() }
        return d

    # 1) strip compile/DataParallel prefixes from ckpt keys
    sd_ckpt_norm = strip_prefixes(sd_ckpt)

    # 2) if model uses DataParallel now, add module. back
    sd_ckpt_norm = add_module_prefix_if_needed(sd_ckpt_norm, sd_model.keys())

    # 3) take only keys that exist AND have the same shape
    to_load = {}
    shape_mismatch = []
    for k, v in sd_ckpt_norm.items():
        if k in sd_model:
            if tuple(v.shape) == tuple(sd_model[k].shape):
                to_load[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(sd_model[k].shape)))

    missing    = sorted([k for k in sd_model.keys() if k not in to_load])
    unexpected = sorted([k for k in sd_ckpt_norm.keys() if k not in sd_model])

    # 4) actually load
    model.load_state_dict(to_load, strict=False)

    # optimizer
    try:
        opt.load_state_dict(ckpt["opt"])
    except Exception as e:
        print(f"[resume] optimizer state not loaded: {e}")

    # scheduler: try native, else align analytically
    try:
        if "sched" in ckpt and hasattr(sched, "load_state_dict") and ckpt["sched"]:
            sched.load_state_dict(ckpt["sched"])
        else:
            _align_warmupcosine_after_resume(sched, int(ckpt.get("global_step", 0)))
    except Exception as e:
        print(f"[resume] scheduler alignment skipped: {e}")

    # 5) report
    total_model = len(sd_model)
    matched = len(to_load)
    print(f"[resume] matched {matched}/{total_model} params "
          f"({matched/total_model:.1%}) | missing={len(missing)} | unexpected={len(unexpected)} | shape_mismatch={len(shape_mismatch)}")
    if missing:
        print("[resume] missing examples:", ", ".join(missing[:8]), ("..." if len(missing) > 8 else ""))
    if unexpected:
        print("[resume] unexpected examples:", ", ".join(unexpected[:8]), ("..." if len(unexpected) > 8 else ""))
    if shape_mismatch:
        ex = ", ".join([f"{k}:{s}->{t}" for k, s, t in shape_mismatch[:4]])
        print("[resume] shape mismatch examples:", ex, ("..." if len(shape_mismatch) > 4 else ""))

    if matched / max(1, total_model) < 0.95:
        print("[resume][WARN] <95% of parameters matched – if this persists after prefix fix, "
              "your model code or YAML args changed since the checkpoint.")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    print(f"[resume] loaded {ckpt_path} | last_epoch={start_epoch-1} | global_step={global_step}")
    return start_epoch, global_step




def main():
    args = parse_args()
    cfg0 = load_yaml(args.config)
    cfg  = apply_overrides(cfg0, args.set)

    # autoload modules to fill registries
    autoload_packages([
        "soundrestorer.models",
        "soundrestorer.tasks",
        "soundrestorer.losses",
    ])

    paths = prepare_run_dirs(cfg)
    print(f"Run dir: {paths['root']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = str(cfg.get("runtime", {}).get("amp", "bfloat16")).lower()
    ch_last = bool(cfg.get("runtime", {}).get("channels_last", True))

    # auto-fallback if bf16 unsupported
    if device.startswith("cuda") and amp in ("bfloat16", "bf16"):
        bf16_ok = False
        try: bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception: bf16_ok = False
        if not bf16_ok:
            print("[runtime] bfloat16 not supported on this GPU – falling back to float16 AMP.")
            amp = "float16"

    # ---- DATA ----
    tr_ds, tr_ld_built, tr_sampler = build_denoise_loader(cfg["data"]["train_manifest"], cfg["data"], train=True)
    va_ds, va_ld_built, _          = build_denoise_loader(cfg["data"]["val_manifest"],   cfg["data"], train=False)

    tr_args = default_loader_args(num_items=len(tr_ds), batch_size=int(cfg["data"]["batch"]),
                                  workers_cfg=cfg["data"].get("workers"))
    va_args = default_loader_args(num_items=len(va_ds), batch_size=int(cfg["data"]["batch"]),
                                  workers_cfg=cfg["data"].get("workers"))
    # enforce/override perf flags
    for dct in (tr_args, va_args):
        dct.setdefault("pin_memory", bool(cfg["data"].get("pin_memory", True)))
        dct.setdefault("persistent_workers", bool(cfg["data"].get("persistent_workers", True)))
        if "prefetch_factor" not in dct and isinstance(cfg["data"].get("prefetch_factor", None), int):
            dct["prefetch_factor"] = int(cfg["data"]["prefetch_factor"])

    va_args.update(shuffle=False, drop_last=False)

    if tr_sampler is not None:
        tr_args.update(sampler=tr_sampler, shuffle=False)

    tr_ld = torch.utils.data.DataLoader(tr_ds, **tr_args)
    va_ld = torch.utils.data.DataLoader(va_ds, **va_args)

    # ---- MODEL ----
    model = MODELS.build(cfg["model"]["name"], **cfg["model"].get("args", {})).float().to(device)

    # Optional head init based on mask variant (SKIP if resuming to avoid overwriting loaded head)
    from soundrestorer.models.init_utils import init_head_for_mask
    want_resume = args.resume is not None and len(str(args.resume)) > 0
    if not want_resume:
        task_mask = cfg.get("task", {}).get("args", {}).get("mask_variant", "plain")
        init_head_for_mask(model, mask_variant=task_mask)

    # ---- TASK ----
    from soundrestorer.core.registry import TASKS
    task = TASKS.build(cfg["task"]["name"], **cfg["task"].get("args", {}))

    # ---- LOSSES ----
    loss_cfg = cfg.get("losses", cfg.get("loss", {}))
    loss_fn = ComposedLoss(loss_cfg["items"])

    # ---- OPT / SCHED ----
    opt_cfg = cfg["optim"]
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("wd", 1e-4))

    sig = inspect.signature(torch.optim.AdamW.__init__).parameters
    use_fused = ("fused" in sig) and torch.cuda.is_available()
    use_foreach = ("foreach" in sig)

    param_groups = _adamw_param_groups(model, wd)
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)  # WD in param_groups

    if use_fused:
        opt = torch.optim.AdamW(param_groups, fused=True, **kwargs)
        print("[optim] AdamW fused=True")
    elif use_foreach:
        opt = torch.optim.AdamW(param_groups, foreach=True, **kwargs)
        print("[optim] AdamW foreach=True")
    else:
        opt = torch.optim.AdamW(param_groups, **kwargs)
        print("[optim] AdamW plain")

    steps_per_epoch = max(1, len(tr_ld))
    grad_accum = int(opt_cfg.get("grad_accum", 1))
    eff_steps_per_epoch = (steps_per_epoch + grad_accum - 1) // grad_accum
    total_steps = int(cfg["train"]["epochs"]) * eff_steps_per_epoch
    min_factor = float(opt_cfg.get("lr_min_factor", 0.3))
    sched = WarmupCosine(opt, total_steps=total_steps,
                         warmup=int(opt_cfg.get("warmup_steps", 200)),
                         min_factor=min_factor)

    # ---- RESUME (load weights/opt, align sched, and possibly compile after) ----
    start_epoch = 1
    global_step = 0
    if want_resume:
        ckpt_path = _find_latest_ckpt(args.resume)
        start_epoch, global_step = _load_resume(ckpt_path, model, opt, sched, device=device)

    # (compile AFTER loading weights)
    model, compiled, compiled_desc = maybe_compile_model(model, cfg.get("runtime", {}))

    # ---- CALLBACKS ----
    ema_decay = float(cfg.get("train", {}).get("ema", 0.0) or 0.0)
    ema_helper = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    cbs = [ConsoleLogger()]
    if ema_helper is not None:
        cbs.append(EmaUpdateCallback(ema_helper))
        cbs.append(EmaEvalSwap(ema_obj=ema_helper, enable=True))

    hm = cfg.get("hard_mining", cfg["data"].get("hard_mining", {}))
    if hm.get("enable", False):
        cbs.append(HardMiningCallback(
            dataset=tr_ds, sampler=tr_sampler,
            start_epoch=hm.get("start_epoch", 3),
            ema=hm.get("ema", 0.9),
            top_frac=hm.get("top_frac", 0.3),
            boost=hm.get("boost", 4.0),
            baseline=hm.get("baseline", 1.0),
        ))

    cbs.append(CheckpointCallback(paths["checkpoints"], every=int(cfg["train"].get("save_every", 1))))

    cb_cfg = cfg.get("callbacks", {})

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

    # ensure min SNR on train batches (rescue too-clean items)
    if cb_cfg.get("ensure_min_snr", {}).get("enable", False):
        em = cb_cfg["ensure_min_snr"]
        from soundrestorer.core.callbacks import EnsureMinSNRCallback
        cbs.append(EnsureMinSNRCallback(
            sr=int(cfg["data"]["sr"]),
            min_snr_db=float(em.get("min_snr_db", 25.0)),
            snr_min=float(em.get("snr_min", 4.0)),
            snr_max=float(em.get("snr_max", 20.0)),
            out_peak=float(cfg["data"].get("out_peak", 0.98)),
            train_only=bool(em.get("train_only", True)),
        ))

    if cb_cfg.get("data_audit", {}).get("enable", False):
        da = cb_cfg["data_audit"]
        from soundrestorer.core.callbacks import DataAuditCallback
        cbs.append(DataAuditCallback(
            out_dir=paths["logs"],
            sr=int(cfg["data"]["sr"]),
            first_epochs=int(da.get("first_epochs", 1)),
            max_batches=int(da.get("max_batches", 2)),
            max_items=int(da.get("max_items", 8)),
            silence_rms_db=float(da.get("silence_rms_db", -60.0)),
            save_audio=bool(da.get("save_audio", True)),
            save_csv=bool(da.get("save_csv", True)),
        ))

    if cb_cfg.get("audio_debug", {}).get("enable", False):
        ad = cb_cfg["audio_debug"]
        cbs.append(AudioDebugCallback(
            out_dir=paths["logs"], val_dataset=va_ds, task=task, sr=int(cfg["data"]["sr"]),
            every=int(ad.get("every", 5)), num_items=int(ad.get("num_items", 3))
        ))

    if cb_cfg.get("curriculum", {}).get("enable", False):
        cur = cb_cfg["curriculum"]
        cbs.append(CurriculumCallback(
            loss_fn=loss_fn,
            task=task,
            dataset=tr_ds,
            snr_stages=cur.get("snr_stages", []),
            sisdr=cur.get("sisdr", {}),
            mask_limit=cur.get("mask_limit", {}),
            mask_variant=cur.get("mask_variant", []),
        ))



    if cb_cfg.get("best_ckpt", {}).get("enable", False):
        bc = cb_cfg["best_ckpt"]
        cbs.append(BestCheckpointCallback(
            out_dir=paths["checkpoints"], k=int(bc.get("top_k", 3)),
            monitor=bc.get("monitor", "val_loss"), mode=bc.get("mode", "min")
        ))

    if cb_cfg.get("early_stop", {}).get("enable", False):
        es = cb_cfg["early_stop"]
        cbs.append(EarlyStoppingCallback(
            patience=int(es.get("patience", 10)),
            min_delta=float(es.get("min_delta", 0.0)),
            monitor=es.get("monitor", "val_loss"),
            mode=es.get("mode", "min")
        ))

    if cb_cfg.get("epoch_seed", {}).get("enable", True):
        cbs.append(EpochSeedCallback([tr_ds, va_ds]))

    # ---- TRAINER ----
    trainer = Trainer(
        model=model, task=task, loss_fn=loss_fn, optimizer=opt, scheduler=sched,
        device=device, amp=amp, grad_accum=grad_accum,
        grad_clip=float(opt_cfg.get("grad_clip", 0.0)),
        channels_last=ch_last, callbacks=cbs
    )
    trainer.runtime = cfg.get("runtime", {})
    trainer.runtime["compile_desc"] = "on" if compiled else "off"
    trainer.compiled = bool(compiled)
    trainer.guard_cfg = cfg.get("guard", {})
    trainer.debug_cfg = cfg.get("debug", {})
    # NOTE: your YAML uses either 'losses' or legacy 'loss' for these
    trainer.train_trim_frac = float(cfg.get("losses", cfg.get("loss", {})).get("train_trim_frac", 0.10))
    trainer.train_loss_clip = float(cfg.get("losses", cfg.get("loss", {})).get("train_loss_clip", 0.0))
    trainer.val_trim_frac   = float(cfg.get("losses", cfg.get("loss", {})).get("val_trim_frac", 0.05))

    # ---- RESUME: set starting epoch & global step on trainer state ----
    if want_resume:
        trainer.state.epoch = int(start_epoch)
        trainer.state.global_step = int(global_step)

    # ---- GO ----
    trainer.fit(tr_ld, va_ld, epochs=int(cfg["train"]["epochs"]), ckpt_saver=None)


if __name__ == "__main__":
    main()
