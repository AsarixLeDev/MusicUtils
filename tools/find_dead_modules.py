#!/usr/bin/env python3
# tools/find_dead_modules.py
"""
Scan repo for Python files and report modules that are not imported by
any non-test/tool module (best-effort, static parsing).
Excludes top-level 'scripts/', 'tests/', 'tools/' as import *consumers*,
but still checks what they import.

Usage: python tools/find_dead_modules.py
"""

import ast, sys
from pathlib import Path

EXCLUDE_CONSUMERS_TOP = {"scripts", "tests", "tools"}  # treat these as non-consumers
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"}
ROOT = Path(__file__).resolve().parent.parent

def is_python_file(p: Path) -> bool:
    return p.suffix == ".py" and all(seg not in EXCLUDE_DIRS for seg in p.parts)

def module_name_from_path(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)

def parse_imports(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return set()
    try:
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods

def main():
    py_files = [p for p in ROOT.rglob("*.py") if is_python_file(p)]
    modules = {module_name_from_path(p): p for p in py_files}

    # consumers: all .py not in scripts/tests/tools
    consumers = []
    for p in py_files:
        parts = p.relative_to(ROOT).parts
        if parts and parts[0] in EXCLUDE_CONSUMERS_TOP:
            continue
        consumers.append(p)

    imported = set()
    for p in consumers:
        mods = parse_imports(p)
        for m in mods:
            imported.add(m)

    # A module is probably unused if:
    #  - it's under our package root ('soundrestorer')
    #  - and its top-level module isn't imported by any consumer
    dead = []
    for mod, path in modules.items():
        if not mod.startswith("soundrestorer"):
            continue
        top = mod.split(".")[1] if "." in mod else mod
        # if nothing imports its top-level package, might still be imported dynamically
        if top not in imported:
            dead.append(path)

    print("\n[find-dead-modules] Candidates (verify before deletion):")
    for p in sorted(dead):
        print(" -", p.relative_to(ROOT))

    print("\nNote:")
    print("  * This is a static scan; dynamic imports (plugins/registries) are not visible.")
    print("  * Check runtime logs (e.g., '[plugins] loaded ...') and grep for these files before deleting.")
    print("  * Typical safe deletions after migration: core/callbacks.py, core/utils.py, losses/sisdr_loss.py, etc.")

if __name__ == "__main__":
    main()
