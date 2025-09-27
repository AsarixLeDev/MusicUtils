from __future__ import annotations

import copy
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    import yaml  # pip install pyyaml
except Exception:  # fallback: minimal YAML via JSON if needed
    yaml = None


# ------------------ dict helpers ------------------

def _deep_get(d: Dict[str, Any], key: str, default=None):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _deep_set(d: Dict[str, Any], key: str, value: Any):
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_dotlist(dotlist: List[str]) -> Dict[str, Any]:
    """
    Convert ["a.b=1", "x.y=z"] → {"a":{"b":1}, "x":{"y":"z"}}
    Attempts to type-cast ints/floats/bools.
    """
    out: Dict[str, Any] = {}
    for item in dotlist:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        val = val.strip()
        # try to cast
        if re.fullmatch(r"-?\d+", val):
            cast = int(val)
        elif re.fullmatch(r"-?\d+\.\d*", val):
            cast = float(val)
        elif val.lower() in ("true", "false"):
            cast = (val.lower() == "true")
        else:
            cast = val
        _deep_set(out, key.strip(), cast)
    return out


# ------------------ YAML / JSON I/O ------------------

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text)
    else:
        # last resort: accept JSON if YAML unavailable
        try:
            return json.loads(text)
        except Exception as e:
            raise RuntimeError("PyYAML not installed and config is not valid JSON") from e


def save_yaml(path: Union[str, Path], data: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ------------------ run directories ------------------

def _timestamp_id(prefix: str = "den") -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{prefix}"


def prepare_run_dirs(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """
    Ensure run folders exist and fill in cfg.paths.* with concrete absolute paths.
    Returns (cfg_updated, paths).
    """
    paths = cfg.get("paths", {})
    run_root = Path(paths.get("run_root", "runs/denoiser"))
    run_id = paths.get("run_id", "") or _timestamp_id("den")
    run_dir = run_root / run_id

    ckpt = paths.get("checkpoints", "")
    graphs = paths.get("graphs", "")
    audio_tests = paths.get("audio_tests", "")

    ckpt_dir = Path(ckpt) if ckpt else run_dir / "checkpoints"
    graphs_dir = Path(graphs) if graphs else run_dir / "graphs"
    audio_dir = Path(audio_tests) if audio_tests else run_dir / "audio_tests"

    for d in (run_dir, ckpt_dir, graphs_dir, audio_dir):
        d.mkdir(parents=True, exist_ok=True)

    # snapshot config
    cfg_snapshot = copy.deepcopy(cfg)
    cfg_snapshot.setdefault("paths", {})
    cfg_snapshot["paths"].update({
        "run_root": str(run_root),
        "run_id": run_id,
        "checkpoints": str(ckpt_dir),
        "graphs": str(graphs_dir),
        "audio_tests": str(audio_dir),
        "run_dir": str(run_dir),
    })
    save_yaml(run_dir / "config_snapshot.yaml", cfg_snapshot)

    # return updated cfg and resolved paths
    cfg["paths"] = cfg_snapshot["paths"]
    return cfg, {
        "run_dir": run_dir,
        "checkpoints": ckpt_dir,
        "graphs": graphs_dir,
        "audio_tests": audio_dir,
    }


# ------------------ public API ------------------

def load_and_prepare(config_path: Union[str, Path], overrides: List[str] | None = None) -> Tuple[
    Dict[str, Any], Dict[str, Path]]:
    """
    Load YAML config, apply --set overrides (dotlist), create run dirs, save snapshot.
    """
    base = load_yaml(config_path)
    ov = parse_dotlist(overrides or [])
    cfg = deep_update(base, ov)
    return prepare_run_dirs(cfg)
