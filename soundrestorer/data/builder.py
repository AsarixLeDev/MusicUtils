import math, torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from .dataset_denoise import DenoiseDatasetWrapper, make_denoise_config
from ..mining.hard_miner import MutableWeightedSampler

# Windows-safe, top-level:
def seed_worker(worker_id: int):
    import random, numpy as np, torch
    info = torch.utils.data.get_worker_info()
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed); np.random.seed(seed)
    try:
        ds = info.dataset
        if hasattr(ds, "_rng"):
            ds._rng = np.random.RandomState(seed)
    except Exception:
        pass

def build_denoise_loader(manifest: str, ds_cfg: dict, train: bool):
    cfg = make_denoise_config(ds_cfg)
    ds = DenoiseDatasetWrapper(manifest, cfg, return_meta=False)

    batch = int(ds_cfg.get("batch", 8))
    workers = int(ds_cfg.get("workers", 2))
    prefetch = int(ds_cfg.get("prefetch_factor", 2))

    sampler = None
    if train and ds_cfg.get("hard_mining", {}).get("enable", False):
        num_samples = int(math.ceil(len(ds) / batch)) * batch
        sampler = MutableWeightedSampler(torch.ones(len(ds), dtype=torch.double), num_samples, replacement=True)

    gen = torch.Generator()
    gen.manual_seed(int(ds_cfg.get("seed", 0)))

    kwargs = dict(
        dataset=ds, batch_size=batch, drop_last=train,
        shuffle=(train and sampler is None),
        num_workers=workers, pin_memory=True,
        persistent_workers=(workers > 0), sampler=sampler,
        worker_init_fn=seed_worker, generator=gen,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch

    ld = DataLoader(**kwargs)
    return ds, ld, sampler
