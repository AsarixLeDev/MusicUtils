from soundrestorer.callbacks.callbacks import Callback


class EpochSeedCallback(Callback):
    def __init__(self, datasets):
        self.datasets = [d for d in (datasets if isinstance(datasets, (list, tuple)) else [datasets]) if d is not None]

    def on_epoch_start(self, state=None, **_):
        seed = 1234 + state.epoch
        import numpy as np, random as _random, torch as _torch
        _random.seed(seed);
        np.random.seed(seed);
        _torch.manual_seed(seed)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(state.epoch)