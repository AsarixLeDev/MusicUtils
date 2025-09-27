from typing import Dict, List, Any
import math, torch
from torch.utils.data import Sampler

class MutableWeightedSampler(Sampler[int]):
    def __init__(self, weights: torch.Tensor, num_samples: int, replacement: bool = True):
        self.weights = weights.clone().to(dtype=torch.double)
        self.num_samples = int(num_samples); self.replacement = bool(replacement)
    def set_weights(self, weights: torch.Tensor):
        self.weights = weights.clone().to(dtype=torch.double)
    def __iter__(self):
        idx = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(idx.tolist())
    def __len__(self): return self.num_samples

class HardMiner:
    def __init__(self, ema=0.9, top_frac=0.3, boost=4.0, baseline=1.0):
        self.ema=float(ema); self.top_frac=float(top_frac); self.boost=float(boost); self.base=float(baseline)
        self.stats: Dict[str,float] = {}
    def update_batch(self, ids, per_sample_losses):
        if ids is None: return
        vals = torch.as_tensor(per_sample_losses).detach().reshape(-1).float().cpu().tolist()
        if isinstance(ids, (list,tuple)):
            flat_ids = []
            for it in ids:
                if isinstance(it, (list,tuple)): flat_ids.extend([str(x) for x in it])
                else: flat_ids.append(str(it))
        else:
            flat_ids = [str(ids)] * len(vals)
        for k,v in zip(flat_ids, vals):
            if not math.isfinite(v): continue
            self.stats[k] = self.ema*self.stats.get(k,v) + (1.0-self.ema)*v
    def make_weights(self, dataset) -> torch.Tensor:
        ids = [dataset._rec_id_from_record(r) for r in dataset.records]
        arr = [self.stats.get(i, 0.0) for i in ids]
        k = max(1, int(len(arr) * self.top_frac))
        thr = sorted(arr)[-k]
        w = [self.base*self.boost if v >= thr else self.base for v in arr]
        return torch.tensor(w, dtype=torch.double)
