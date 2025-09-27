# soundrestorer/core/prefetch.py
from __future__ import annotations
import torch

class CUDAPrefetcher:
    def __init__(self, loader, device: str = "cuda", channels_last: bool = True):
        self.loader = iter(loader)
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream() if device.startswith("cuda") else None
        self.next = None
        self._preload()

    def _to(self, t):
        if t is None: return None
        if isinstance(t, torch.Tensor):
            return t.to(self.device, non_blocking=True)
        return torch.as_tensor(t).to(self.device, non_blocking=True)

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next = None
            return
        if self.stream is None:
            self.next = batch
            return
        with torch.cuda.stream(self.stream):
            if isinstance(batch, (list, tuple)):
                moved = []
                for x in batch:
                    if isinstance(x, dict):
                        moved.append(x)      # meta stays on CPU
                    else:
                        moved.append(self._to(x))
                self.next = tuple(moved)
            else:
                self.next = self._to(batch)

    def next_batch(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        out = self.next
        if out is not None and self.channels_last:
            # best effort: put 4D tensors in channels_last (B,2,F,T)
            def _fmt(u):
                return u.contiguous(memory_format=torch.channels_last) if (
                    isinstance(u, torch.Tensor) and u.dim() == 4) else u
            if isinstance(out, (list, tuple)):
                out = tuple(_fmt(u) for u in out)
            else:
                out = _fmt(out)
        self._preload()
        return out
