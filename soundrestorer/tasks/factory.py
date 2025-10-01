# soundrestorer/tasks/factory.py
from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, Optional, Callable

import torch.nn as nn

# Map legacy name straight to the new task
TASK_ALIASES = {
    "denoise_stft": ("soundrestorer.tasks.denoise_stft_music", "DenoiseSTFTMusic"),
    "denoise_stft_music": ("soundrestorer.tasks.denoise_stft_music", "DenoiseSTFTMusic"),
}


def _import(path: str) -> ModuleType:
    return importlib.import_module(path)


def _pick_ctor(mod: ModuleType) -> Optional[Callable[..., Any]]:
    # Prefer factory functions if present
    for fn in ("create_task", "build_task", "create", "build", "make"):
        c = getattr(mod, fn, None)
        if callable(c):
            return c
    # Fall back to the most likely nn.Module class
    candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
            candidates.append(obj)
    if not candidates:
        return None

    # Prefer denoise/stft/mask names
    def _rank(cls):
        n = cls.__name__.lower()
        score = sum(k in n for k in ("denoise", "stft", "mask", "task", "model"))
        return (-score, len(n))

    candidates.sort(key=_rank)
    return candidates[0]


def _call_ctor_adapt(ctor: Callable[..., Any], args: Dict[str, Any]):
    """
    Call task constructor robustly:
      - If signature has (model, args), pack the remaining kwargs into 'args' dict
      - Otherwise, pass kwargs as-is
    """
    sig = inspect.signature(ctor)
    params = sig.parameters
    # Must have a model either way
    if "model" not in args or args["model"] is None:
        raise TypeError("create_task(...) requires 'model=<nn.Module>'")

    # Case 1: ctor expects (model, args)
    if "model" in params and "args" in params and all(
            (k in ("self", "model", "args") or params[k].kind in (inspect.Parameter.VAR_KEYWORD,
                                                                  inspect.Parameter.VAR_POSITIONAL))
            for k in params
    ):
        model = args["model"]
        cfg = {k: v for k, v in args.items() if k != "model"}
        return ctor(model=model, args=cfg)

    # Case 2: ctor exposes specific kwargs (e.g., n_fft, hop_length, ...)
    try:
        return ctor(**args)
    except TypeError:
        # Last resort: try (model, args) even if signature parse failed above
        model = args["model"]
        cfg = {k: v for k, v in args.items() if k != "model"}
        return ctor(model=model, args=cfg)


def create_task(name_or_cfg: Any, **kwargs) -> nn.Module:
    """
    Accept either:
      - create_task("denoise_stft", model=..., n_fft=..., ...)
      - create_task({"name":"denoise_stft","args":{...}}, model=...)
    """
    # Normalize inputs
    if isinstance(name_or_cfg, dict):
        name = str(name_or_cfg.get("name", "")).strip()
        args = dict(name_or_cfg.get("args") or {})
        args.update(kwargs or {})
    else:
        name = str(name_or_cfg).strip()
        args = dict(kwargs or {})

    # 1) Alias path (your A2 choice)
    if name in TASK_ALIASES:
        mod_path, cls_name = TASK_ALIASES[name]
        mod = _import(mod_path)
        ctor = getattr(mod, cls_name) if cls_name else _pick_ctor(mod)
        if ctor is None:
            raise ImportError(f"{mod_path}: no task factory/class exported")
        return _call_ctor_adapt(ctor, args)

    # 2) Heuristic paths
    for cand in (
            f"soundrestorer.tasks.{name}",
            f"soundrestorer.tasks.{name.replace('-', '_')}",
            name,  # fully qualified
    ):
        try:
            mod = _import(cand)
            ctor = _pick_ctor(mod)
            if ctor is None:
                continue
            return _call_ctor_adapt(ctor, args)
        except Exception:
            continue

    raise ImportError(f"create_task('{name}') could not resolve any task module")
