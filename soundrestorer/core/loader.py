# soundrestorer/core/loader.py
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_worker(worker_id: int):
    # Different RNG stream per worker & epoch
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int | None = None) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed if seed is not None else torch.seed())
    return g


def default_loader_args(num_items: int, batch_size: int, workers_cfg: int | None = None):
    # Sensible defaults for Windows/NVMe: a few workers, deeper prefetch
    cpu = max(1, (os.cpu_count() or 4))
    nw = workers_cfg if workers_cfg is not None else min(6, max(2, cpu // 3))
    pf = 4 if nw > 0 else None
    return dict(
        batch_size=batch_size,
        shuffle=True, drop_last=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(pf if nw > 0 else None),
        worker_init_fn=seed_worker,
        generator=make_generator(),
    )
