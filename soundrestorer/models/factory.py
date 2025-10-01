# soundrestorer/models/factory.py
from __future__ import annotations
import importlib, inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Callable
import torch.nn as nn

# Friendly aliases so config names resolve to your real modules/classes
MODEL_ALIASES = {
    # name_in_config : (module_path, class_name)
    "complex_unet_auto": ("soundrestorer.models.auto", "AutoComplexUNet"),
    "complex_unet":      ("soundrestorer.models.complex_unet", "ComplexUNetWrapper"),
    "complex_unet_lstm": ("soundrestorer.models.complex_unet_lstm", "ComplexUNetLSTM"),
    "music_unet":        ("soundrestorer.models.music_unet", "MusicUNet"),
}

def _import_module(path: str) -> ModuleType:
    return importlib.import_module(path)

def _ctor_from_module(mod: ModuleType) -> Optional[Callable[..., Any]]:
    # Prefer a factory function if present
    for fn in ("create_model", "build_model", "create", "build", "make"):
        ctor = getattr(mod, fn, None)
        if callable(ctor):
            return ctor
    # Else pick an nn.Module subclass defined here
    candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
            candidates.append(obj)
    if not candidates:
        return None
    # simple ranking: model-ish names first
    def _rank(cls):
        n = cls.__name__.lower()
        score = sum(k in n for k in ("unet", "denoise", "denoiser", "net", "model"))
        return (-score, len(n))
    candidates.sort(key=_rank)
    return candidates[0]

def create_model(name_or_cfg: Any, **kwargs) -> nn.Module:
    """
    Accepts either:
      - create_model("complex_unet_auto", base=48, ...)
      - create_model({"name":"complex_unet_auto","args":{"base":48,...}})
    """
    # Normalize args
    if isinstance(name_or_cfg, dict):
        name = str(name_or_cfg.get("name", "")).strip()
        args = dict(name_or_cfg.get("args") or {})
        # kwargs override cfg args if both given
        args.update(kwargs or {})
    else:
        name = str(name_or_cfg).strip()
        args = dict(kwargs or {})

    # 1) Fast path via alias table
    if name in MODEL_ALIASES:
        mod_path, cls_name = MODEL_ALIASES[name]
        mod = _import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls(**args)

    # 2) Heuristic tries: soundrestorer.models.<name> etc.
    attempts = []
    for cand in (
        f"soundrestorer.models.{name}",
        f"soundrestorer.models.{name.replace('-', '_')}",
        name,  # fully-qualified path case
    ):
        try:
            mod = _import_module(cand)
        except Exception as e:
            attempts.append(f"{cand}: {type(e).__name__}: {e}")
            continue
        ctor = _ctor_from_module(mod)
        if ctor is None:
            attempts.append(f"{cand}: no factory or nn.Module found")
            continue
        try:
            return ctor(**args)  # works for function or class
        except TypeError as e:
            sig = getattr(ctor, "__name__", "<class>")
            attempts.append(f"{cand}.{sig} signature mismatch: {e}")
        except Exception as e:
            sig = getattr(ctor, "__name__", "<class>")
            attempts.append(f"{cand}.{sig} failed: {type(e).__name__}: {e}")

    msg = "; ".join(attempts) if attempts else "no attempts made"
    raise ImportError(f"create_model('{name}') could not resolve. Tried: {msg}")
