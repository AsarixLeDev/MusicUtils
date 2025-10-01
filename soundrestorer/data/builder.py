# soundrestorer/data/builder.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

try:
    from .music_denoise_dataset import MusicDenoiseDataset
except Exception:
    MusicDenoiseDataset = None  # type: ignore
try:
    from .dataset_denoise import DenoiseDataset
except Exception:
    DenoiseDataset = None  # type: ignore


def _peek_kind(manifest: Path) -> str:
    with manifest.open("r", encoding="utf-8") as f:
        for _ in range(16):
            line = f.readline().strip()
            if not line: continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                if "stems" in row: return "music"
                if "clean" in row: return "pair"
    return "pair"


def _pick_dataset(kind: str):
    if kind == "music" and MusicDenoiseDataset is not None:
        return MusicDenoiseDataset
    return DenoiseDataset


def _inst(cls, manifest: Path, split: str, dcfg: Dict[str, Any]):
    try:
        return cls(str(manifest), dcfg)  # many datasets accept (manifest_path, **cfg)
    except TypeError:
        # try common kw names for the path
        for key in ("manifest", "manifest_path", "jsonl", "path"):
            try:
                return cls(**{key: str(manifest)}, **dcfg)
            except TypeError:
                continue
        raise


def _mk_loader(ds, dcfg: Dict[str, Any], train: bool) -> DataLoader:
    bs = int(dcfg.get("batch", 12))
    nw = int(dcfg.get("workers", 0))
    pin = bool(dcfg.get("pin_memory", True))
    args = dict(batch_size=bs, shuffle=train, num_workers=nw, pin_memory=pin)
    pf = dcfg.get("prefetch_factor", None)
    if pf is not None and nw > 0:
        args["prefetch_factor"] = int(pf)
        args["persistent_workers"] = bool(dcfg.get("persistent_workers", False))
    collate = getattr(ds, "collate_fn", None)
    if callable(collate):
        args["collate_fn"] = collate
    return DataLoader(ds, **args)


def build_loaders(cfg: Dict[str, Any]):
    dcfg = cfg if "data" not in cfg else cfg["data"]
    tr_manifest = Path(dcfg["train_manifest"])
    va_manifest = Path(dcfg.get("val_manifest", "")) if dcfg.get("val_manifest") else None

    tr_kind = _peek_kind(tr_manifest)
    tr_cls = _pick_dataset(tr_kind)
    if tr_cls is None:
        raise ImportError(f"No dataset class available for kind='{tr_kind}'")
    tr_ds = _inst(tr_cls, tr_manifest, split="train", dcfg=dcfg)
    tr_ld = _mk_loader(tr_ds, dcfg, train=True)

    va_ld = None
    if va_manifest and va_manifest.exists():
        va_kind = _peek_kind(va_manifest)
        va_cls = _pick_dataset(va_kind) or tr_cls
        va_ds = _inst(va_cls, va_manifest, split="val", dcfg=dcfg)
        va_ld = _mk_loader(va_ds, dcfg, train=False)

    info = dict(
        train=dict(kind=tr_kind, n=len(tr_ds)),
        val=dict(kind=(va_manifest and va_manifest.exists() and va_kind) or None,
                 n=(va_ld and len(va_ld.dataset)) or 0),
        sr=int(dcfg.get("sr", 48000)),
    )
    return tr_ld, va_ld, info
