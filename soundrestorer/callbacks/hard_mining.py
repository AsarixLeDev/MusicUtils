from typing import Optional

from soundrestorer.callbacks.callbacks import Callback
from ..mining.hard_miner import HardMiner, MutableWeightedSampler


class HardMiningCallback(Callback):
    """
    Uses per-sample proxy from task (or computes L1) to update a mutable weighted sampler.
    """

    def __init__(self, dataset, sampler: Optional[MutableWeightedSampler],
                 start_epoch=3, ema=0.9, top_frac=0.3, boost=4.0, baseline=1.0):
        self.dataset = dataset
        self.sampler = sampler
        self.start_epoch = int(start_epoch)
        self.miner = HardMiner(ema=ema, top_frac=top_frac, boost=boost, baseline=baseline)

    def on_batch_end(self, per_sample=None, outputs=None, **_):
        if per_sample is None:
            return
        ids = outputs.get("ids", None) if isinstance(outputs, dict) else None
        self.miner.update_batch(ids, per_sample)

    def on_epoch_end(self, state=None, **_):
        if not self.sampler or state.epoch < self.start_epoch:
            return
        w = self.miner.make_weights(self.dataset)
        self.sampler.set_weights(w)
        print("[hard-mining] sampler weights updated.")