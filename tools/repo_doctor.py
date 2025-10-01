#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Doctor: scans a SoundRestorer repo for wiring/exports mismatches.
- No writes, no imports that would crash training (imports are guarded).
- Prints a human-readable report and can also dump JSON (--json-out ...).

Usage:
  python tools/project_doctor.py [--root .] [--config configs/denoiser_flexible_v2.yaml] [--json-out report.json]

Exit code is non-zero if critical items are missing.
"""
from __future__ import annotations
import argparse, json, os, sys, pkgutil, importlib, inspect, ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# ------------------------- small helpers -------------------------

def safe_import(mod: str):
    try:
        m = importlib.import_module(mod)
        return True, m, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def has_attr(mod, name: str) -> Tuple[bool, Optional[Any]]:
    try:
        obj = getattr(mod, name)
        return True, obj
    except Exception:
        return False, None

def find_repo_root(root_hint: Path) -> Path:
    # prefer a directory that contains 'soundrestorer' package
    root = root_hint.resolve()
    if (root / "soundrestorer").is_dir():
        return root
    # walk up
    cur = root
    for _ in range(5):
        if (cur / "soundrestorer").is_dir():
            return cur
        cur = cur.parent
    return root

def list_py_files(base: Path) -> List[Path]:
    return [p for p in base.rglob("*.py") if ("__pycache__" not in str(p))]

def module_path_from_file(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    return ".".join(parts)

def nn_classes(mod) -> List[str]:
    import torch.nn as nn  # type: ignore
    return [name for name, obj in inspect.getmembers(mod, inspect.isclass) if issubclass(obj, nn.Module)]

def parse_functions(file: Path) -> List[str]:
    try:
        src = file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
        return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    except Exception:
        return []

def yaml_load(path: Path) -> Dict[str, Any]:
    import yaml  # lazy import
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ------------------------- scans -------------------------

def scan_factories() -> Dict[str, Any]:
    report = {}
    # models.factory.create_model
    ok, m, err = safe_import("soundrestorer.models.factory")
    report["models.factory"] = {"import_ok": ok, "error": err}
    if ok:
        exists, fn = has_attr(m, "create_model")
        report["models.factory"]["create_model"] = exists

    # tasks.factory.create_task
    ok, m, err = safe_import("soundrestorer.tasks.factory")
    report["tasks.factory"] = {"import_ok": ok, "error": err}
    if ok:
        exists, fn = has_attr(m, "create_task")
        report["tasks.factory"]["create_task"] = exists

    # losses.composed.build_losses
    ok, m, err = safe_import("soundrestorer.losses.composed")
    report["losses.composed"] = {"import_ok": ok, "error": err}
    if ok:
        exists, fn = has_attr(m, "build_losses")
        report["losses.composed"]["build_losses"] = exists

    # data.builder(s).build_loaders
    tried = []
    for target in ("soundrestorer.data.builder", "soundrestorer.data.builders"):
        ok, m, err = safe_import(target)
        tried.append({"module": target, "ok": ok, "error": err})
        if ok:
            report["data.builder"] = {"module": target}
            exists, fn = has_attr(m, "build_loaders")
            report["data.builder"]["build_loaders"] = exists
            break
    else:
        report["data.builder"] = {"module": None, "build_loaders": False, "tried": tried}
    return report

def scan_models_and_tasks(repo_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"models": [], "tasks": []}
    # models
    for m in pkgutil.iter_modules([str(repo_root / "soundrestorer" / "models")]):
        mod_name = f"soundrestorer.models.{m.name}"
        ok, mod, err = safe_import(mod_name)
        rec = {"module": mod_name, "import_ok": ok, "error": err, "nn_classes": []}
        if ok:
            rec["nn_classes"] = nn_classes(mod)
        out["models"].append(rec)
    # tasks
    for m in pkgutil.iter_modules([str(repo_root / "soundrestorer" / "tasks")]):
        mod_name = f"soundrestorer.tasks.{m.name}"
        ok, mod, err = safe_import(mod_name)
        rec = {"module": mod_name, "import_ok": ok, "error": err, "nn_classes": []}
        if ok:
            rec["nn_classes"] = nn_classes(mod)
        out["tasks"].append(rec)
    return out

HELPER_NAMES = {"to_mono", "_to_mono", "si_sdr_db", "rms_db", "hann_window"}

def scan_helper_duplicates(repo_root: Path) -> Dict[str, Any]:
    dup: Dict[str, List[Dict[str, Any]]] = {n: [] for n in HELPER_NAMES}
    for py in list_py_files(repo_root / "soundrestorer"):
        fns = parse_functions(py)
        for h in HELPER_NAMES:
            if h in fns:
                dup[h].append({"module": module_path_from_file(repo_root, py), "file": str(py)})
    return {k: v for k, v in dup.items() if v}

def check_inits(repo_root: Path) -> List[str]:
    missing = []
    for pkg in ["soundrestorer",
                "soundrestorer/models",
                "soundrestorer/tasks",
                "soundrestorer/losses",
                "soundrestorer/data",
                "soundrestorer/callbacks",
                "soundrestorer/core",
                "soundrestorer/metrics",
                "soundrestorer/utils"]:
        if not (repo_root / pkg / "__init__.py").exists():
            missing.append(str(repo_root / pkg / "__init__.py"))
    return missing

def resolve_config(cfg_path: Optional[Path]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"config": None}
    if not cfg_path or not cfg_path.exists():
        rep["note"] = "config not provided or missing"
        return rep
    cfg = yaml_load(cfg_path)
    rep["config"] = str(cfg_path)
    model_name = (cfg.get("model", {}) or {}).get("name")
    task_name  = (cfg.get("task",  {}) or {}).get("name")
    rep["model.name"] = model_name
    rep["task.name"]  = task_name

    # First try factories (if available)
    def try_factory(factory_mod: str, fn_name: str, key: str, name: Optional[str]) -> Dict[str, Any]:
        res = {"name": name, "factory_module": factory_mod, "factory_fn": fn_name}
        if not name:
            res["ok"] = False; res["error"] = "name missing in config"; return res
        ok, mod, err = safe_import(factory_mod)
        if not ok:
            res["ok"] = False; res["error"] = f"import failed: {err}"; return res
        ok_attr, fn = has_attr(mod, fn_name)
        if not ok_attr:
            res["ok"] = False; res["error"] = f"factory missing: {factory_mod}.{fn_name}"; return res
        # just ask the factory which keys it knows, if it exposes a registry-like dict
        try:
            known = getattr(mod, "REGISTRY", None) or getattr(mod, "MODEL_ZOO", None) or getattr(mod, "TASKS", None)
            if isinstance(known, dict):
                res["known_keys"] = sorted(list(known.keys()))
                res["known_contains"] = name in known
        except Exception:
            pass
        res["ok"] = True
        return res

    rep["model.factory_check"] = try_factory("soundrestorer.models.factory", "create_model", "model", model_name)
    rep["task.factory_check"]  = try_factory("soundrestorer.tasks.factory",  "create_task",  "task",  task_name)

    # Heuristic direct import paths to help debugging
    def try_module_variants(prefix: str, n: Optional[str]) -> List[Dict[str, Any]]:
        if not n: return []
        cands = [
            f"{prefix}.{n}",
            f"{prefix}.{n.replace('-', '_')}",
            f"{prefix}.{n.replace('complex_', '')}",  # crude but useful with "complex_unet_auto"
        ]
        seen = set()
        tries = []
        for c in cands:
            if c in seen: continue
            seen.add(c)
            ok, mod, err = safe_import(c)
            tries.append({"target": c, "ok": ok, "error": err if not ok else None})
        return tries

    rep["model.import_tries"] = try_module_variants("soundrestorer.models", model_name)
    rep["task.import_tries"]  = try_module_variants("soundrestorer.tasks",  task_name)
    return rep

# ------------------------- reporter -------------------------

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repository root (folder that contains soundrestorer/)")
    ap.add_argument("--config", type=str, default="", help="Optional YAML to sanity-check name resolution")
    ap.add_argument("--json-out", type=str, default="", help="Write a JSON report to this path")
    args = ap.parse_args()

    repo_root = find_repo_root(Path(args.root))
    sys.path.insert(0, str(repo_root))  # allow local imports

    # 1) package init sanity
    missing_inits = check_inits(repo_root)
    print_section("1) Package init sanity")
    if missing_inits:
        for p in missing_inits:
            print(f"[MISS] create empty __init__: {p}")
    else:
        print("[OK] all expected __init__.py present")

    # 2) core factories
    print_section("2) Factories & builder wiring")
    factories = scan_factories()
    for k, v in factories.items():
        print(f"- {k}: {v}")

    # 3) models / tasks importability
    print_section("3) Import scan — models")
    mt = scan_models_and_tasks(repo_root)
    for m in mt["models"]:
        ok = "OK " if m["import_ok"] else "ERR"
        print(f"[{ok}] {m['module']}  classes={m['nn_classes']}")
        if not m["import_ok"]:
            print(f"      └─ {m['error']}")
    print_section("3b) Import scan — tasks")
    for t in mt["tasks"]:
        ok = "OK " if t["import_ok"] else "ERR"
        print(f"[{ok}] {t['module']}  classes={t['nn_classes']}")
        if not t["import_ok"]:
            print(f"      └─ {t['error']}")

    # 4) helper duplicates
    print_section("4) Helper duplicates (consider unifying)")
    dups = scan_helper_duplicates(repo_root)
    if not dups:
        print("[OK] no duplicate helper names found")
    else:
        for name, places in dups.items():
            print(f"- {name}: {len(places)} definitions")
            for p in places:
                print(f"    • {p['module']} ({p['file']})")

    # 5) config resolution
    cfg_rep = resolve_config(Path(args.config)) if args.config else {"note": "skipped (no --config)"}
    print_section("5) Config name resolution")
    print(json.dumps(cfg_rep, indent=2))

    # 6) actionable suggestions
    print_section("6) Suggestions")
    suggestions = []
    # factories
    if not factories.get("losses.composed", {}).get("import_ok", False) or not factories.get("losses.composed", {}).get("build_losses", False):
        suggestions.append("Add `build_losses(cfg)` to `soundrestorer/losses/composed.py` that returns a composed loss from cfg['losses']['items'].")
    bldr = factories.get("data.builder", {})
    if not bldr or not bldr.get("build_loaders", False):
        suggestions.append("Ensure `build_loaders(...)` exists. Either implement it in `soundrestorer/data/builder.py` or create `data/builders.py` that re-exports from builder.py for backward compatibility.")
    # tasks hann_window
    # Quick static hint: many denoise_stft failures come from importing hann_window from utils.audio
    suggestions.append("If `denoise_stft` imports `hann_window` from utils.audio, either implement `hann_window` there or switch to `torch.hann_window` in the task.")
    # helper duplicates
    if dups:
        suggestions.append("Unify helpers: keep `to_mono` in utils.audio, `si_sdr_db` in utils.metrics, `rms_db` in utils.metrics; migrate imports accordingly.")
    # __init__ missing
    for p in missing_inits:
        suggestions.append(f"Add empty __init__.py: {p}")
    # model/task name mapping
    if args.config:
        mn = cfg_rep.get("model.name")
        tn = cfg_rep.get("task.name")
        mf = cfg_rep.get("model.factory_check", {})
        tf = cfg_rep.get("task.factory_check",  {})
        if mn and (not mf.get("ok") or not mf.get("known_contains", True)):
            suggestions.append(f"Map '{mn}' in models.factory (e.g., REGISTRY['{mn}']=Class) or rename the file to match import heuristics.")
        if tn and (not tf.get("ok") or not tf.get("known_contains", True)):
            suggestions.append(f"Map '{tn}' in tasks.factory or rename task module accordingly.")
    if suggestions:
        for s in suggestions:
            print(f"- {s}")
    else:
        print("[OK] No obvious suggestions; wiring looks consistent.")

    # 7) write JSON if requested
    if args.json_out:
        out = {
            "repo_root": str(repo_root),
            "factories": factories,
            "models_and_tasks": mt,
            "duplicates": dups,
            "missing_inits": missing_inits,
            "config_resolution": cfg_rep,
            "suggestions": suggestions,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[WROTE] JSON report -> {args.json_out}")

    # exit code: 1 if critical items are missing
    critical = []
    if not factories.get("models.factory", {}).get("create_model", False): critical.append("models.factory.create_model")
    if not factories.get("tasks.factory",  {}).get("create_task",  False): critical.append("tasks.factory.create_task")
    if not factories.get("losses.composed", {}).get("build_losses", False): critical.append("losses.composed.build_losses")
    if not bldr or not bldr.get("build_loaders", False): critical.append("data.builder(s).build_loaders")
    if critical:
        print("\n[EXIT] Critical missing items:\n  - " + "\n  - ".join(critical))
        sys.exit(2)
    else:
        print("\n[EXIT] OK")
        sys.exit(0)

if __name__ == "__main__":
    main()
