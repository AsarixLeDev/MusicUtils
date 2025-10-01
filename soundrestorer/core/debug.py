# soundrestorer/core/debug.py
from __future__ import annotations
import time, math, torch
from typing import Dict, Any, Optional

def param_counts(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train, "frozen": total - train}

def fmt_compact_comps(d: dict, order=("mrstft","mel_l1","l1_wave","phase_cosine","highband_l1","sisdr_ratio","energy_anchor")):
    parts = []
    for k in order:
        if k in d:
            parts.append(f"{k.replace('_','')}={d[k]:.3f}")
    return " ".join(parts)

def fmt_skip_reasons(skip_reasons: dict, top=4):
    if not skip_reasons:
        return ""
    it = sorted(skip_reasons.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return " | skips: " + ", ".join(f"{k} ×{v}" for k,v in it)


def short_device_summary(model: torch.nn.Module, runtime: Dict[str, Any]) -> str:
    p = next(model.parameters())
    dev = str(p.device)
    amp = runtime.get("amp", "bfloat16")
    ch  = bool(runtime.get("channels_last", True))
    return f"device={dev} | amp={amp} | channels_last={ch}"

def lr_of(optimizer: torch.optim.Optimizer) -> float:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return float("nan")

def cuda_mem_mb() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    alloc = torch.cuda.memory_allocated() / (1024**2)
    resv  = torch.cuda.memory_reserved() / (1024**2)
    return f"{alloc:.0f}/{resv:.0f} MiB"
