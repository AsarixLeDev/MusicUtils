# -*- coding: utf-8 -*-
# soundrestorer/utils/torch_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


# ------------- device / amp --------------

def move_to(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(move_to(t, device) for t in x)
    if isinstance(x, dict):
        return {k: move_to(v, device) for k, v in x.items()}
    return x


def autocast_from_amp(amp: str):
    """
    amp: 'bfloat16' | 'float16' | 'off'
    """
    a = (amp or "off").lower()
    if a in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if a in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def need_grad_scaler(amp: str) -> bool:
    return (amp or "").lower() in ("fp16", "float16", "half")


def set_channels_last(m: nn.Module, enable: bool = True):
    if enable:
        m.to(memory_format=torch.channels_last)


# ------------- checkpoint helpers --------------

def strip_state_dict_prefixes(sd: Dict[str, torch.Tensor], prefixes=("module.", "_orig_mod.")) -> Dict[
    str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_state_dict_loose(module: nn.Module, sd: Dict[str, torch.Tensor], tag: str = "model"):
    current = module.state_dict()
    incoming = strip_state_dict_prefixes(sd)
    matched, missing, unexpected, shape_mismatch = 0, [], [], 0
    for k, v in incoming.items():
        if k in current:
            if current[k].shape != v.shape:
                shape_mismatch += 1
            else:
                current[k] = v
                matched += 1
        else:
            unexpected.append(k)
    for k in current.keys():
        if k not in incoming:
            missing.append(k)
    module.load_state_dict(current, strict=False)
    print(f"[resume] {tag}: matched {matched}/{len(current)} params "
          f"({matched / max(1, len(current)) * 100:.1f}%) | missing={len(missing)} | "
          f"unexpected={len(unexpected)} | shape_mismatch={shape_mismatch}")
    if missing:
        ex = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"[resume] missing examples: {ex}")
    if unexpected:
        ex = ", ".join(unexpected[:8]) + (" ..." if len(unexpected) > 8 else "")
        print(f"[resume] unexpected examples: {ex}")


def latest_checkpoint(path: str | Path) -> Optional[Path]:
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists():
        return None
    cks = sorted(p.glob("epoch_*.pt"))
    return cks[-1] if cks else None


# ------------- misc --------------

def format_ema(ema_beta: float) -> str:
    return f"{ema_beta:.3f}" if float(ema_beta) > 0 else "OFF"


def set_seed(seed: int):
    import os, random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
