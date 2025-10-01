# soundrestorer/tasks/factory.py
from __future__ import annotations
import importlib, inspect
from types import ModuleType
from typing import Any, Dict, Optional, Callable
import torch.nn as nn

TASK_ALIASES = {
    "denoise_stft":        ("soundrestorer.tasks.denoise_stft",        None),  # prefer legacy if import works
    "denoise_stft_music":  ("soundrestorer.tasks.denoise_stft_music",  "DenoiseSTFTMusic"),
}

def _import(path: str) -> ModuleType:
    return importlib.import_module(path)

def _ctor(mod: ModuleType) -> Optional[Callable[..., Any]]:
    for fn in ("create_task", "build_task", "create", "build", "make"):
        c = getattr(mod, fn, None)
        if callable(c): return c
    # else class
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
            return obj
    return None

def create_task(name_or_cfg: Any, **kwargs) -> nn.Module:
    if isinstance(name_or_cfg, dict):
        name = str(name_or_cfg.get("name", "")).strip()
        args = dict(name_or_cfg.get("args") or {})
        args.update(kwargs or {})
    else:
        name = str(name_or_cfg).strip()
        args = dict(kwargs or {})

    # 1) Try alias match
    if name in TASK_ALIASES:
        mod_path, cls_name = TASK_ALIASES[name]
        try:
            mod = _import(mod_path)
            ctor = getattr(mod, cls_name) if cls_name else _ctor(mod)
            if ctor is None:
                raise ImportError(f"{mod_path}: no task factory/class")
            return ctor(**args)
        except Exception as e:
            # Graceful fallback: if legacy denoise_stft fails (hann_window), try the music variant
            if name == "denoise_stft":
                mod = _import("soundrestorer.tasks.denoise_stft_music")
                return getattr(mod, "DenoiseSTFTMusic")(**args)
            raise

    # 2) Heuristic tries
    for cand in (
        f"soundrestorer.tasks.{name}",
        f"soundrestorer.tasks.{name.replace('-', '_')}",
        name,  # fully-qualified
    ):
        try:
            mod = _import(cand)
            ctor = _ctor(mod)
            if ctor is None:
                continue
            return ctor(**args)
        except Exception:
            continue

    raise ImportError(f"create_task('{name}') could not resolve any task module.")
