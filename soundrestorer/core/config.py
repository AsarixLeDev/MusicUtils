import os, json, yaml, time
from copy import deepcopy
from typing import Any, List, Dict

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _parse_value(v: str):
    # interpret numbers/bools/nulls
    lowered = v.lower()
    if lowered in ("true","false"):
        return lowered == "true"
    if lowered in ("null","none"):
        return None
    try:
        if "." in v: return float(v)
        return int(v)
    except ValueError:
        return v

def apply_overrides(cfg: dict, args: List[str]) -> dict:
    out = deepcopy(cfg)
    for a in (args or []):
        # key.subkey.subsub=value
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        v = _parse_value(v)
        node = out
        keys = k.split(".")
        for kk in keys[:-1]:
            if kk not in node or not isinstance(node[kk], dict):
                node[kk] = {}
            node = node[kk]
        node[keys[-1]] = v
    return out

def prepare_run_dirs(cfg: dict) -> dict:
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg.get("paths", {}).get("run_name", "train")
    root = os.path.join(runs_root, f"{run_id}_{run_name}")
    ckpt = os.path.join(root, "checkpoints")
    logs = os.path.join(root, "logs")
    os.makedirs(ckpt, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    return {"root": root, "checkpoints": ckpt, "logs": logs}
