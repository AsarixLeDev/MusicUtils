import importlib, pkgutil

def autoload_packages(packages):
    """
    Import every submodule under the given packages so registries fill
    via @register decorators.
    """
    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            print(f"[plugins] skip {pkg_name}: {e}")
            continue
        # If it's a module (not a package), importing once is enough
        if not hasattr(pkg, "__path__"):
            continue
        for m in pkgutil.iter_modules(pkg.__path__):
            mod_name = f"{pkg_name}.{m.name}"
            try:
                importlib.import_module(mod_name)
                # print(f"[plugins] loaded {mod_name}")
            except Exception as e:
                print(f"[plugins] failed {mod_name}: {e}")