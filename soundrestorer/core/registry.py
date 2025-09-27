# Minimal registry with decorators

from typing import Callable, Dict, Any

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Any] = {}

    def register(self, name: str = None):
        def deco(obj):
            key = (name or obj.__name__).lower()
            if key in self._store:
                raise KeyError(f"{self.name} registry already has key: {key}")
            self._store[key] = obj
            return obj
        return deco

    def get(self, name: str):
        obj = self._store.get(name.lower())
        if obj is None:
            raise KeyError(f"{self.name} not found: {name}")
        return obj

    def build(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

MODELS     = Registry("model")
DATASETS   = Registry("dataset")
LOSSES     = Registry("loss")
TASKS      = Registry("task")
CALLBACKS  = Registry("callback")
