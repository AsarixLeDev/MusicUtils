from __future__ import annotations
import torch

class CUDAPrefetcher:
    def __init__(self, loader, device: str = "cuda", channels_last: bool = True):
        self.loader = iter(loader)
        self.device = device
        self.channels_last = bool(channels_last)
        self.stream = torch.cuda.Stream() if device.startswith("cuda") else None
        self._next = None
        self._preload()

    def _to(self, t):
        if t is None: return None
        if isinstance(t, torch.Tensor): return t.to(self.device, non_blocking=True)
        return t

    def _fmt(self, t):
        if self.channels_last and isinstance(t, torch.Tensor) and t.dim() == 4:
            return t.contiguous(memory_format=torch.channels_last)
        return t

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self._next = None
            return
        if self.stream is None:
            self._next = batch
            return
        with torch.cuda.stream(self.stream):
            if isinstance(batch, (list, tuple)):
                moved = []
                for x in batch:
                    if isinstance(x, dict): moved.append(x)
                    elif isinstance(x, (list, tuple)):
                        moved.append(type(x)(self._to(xx) for xx in x))
                    else: moved.append(self._to(x))
                self._next = type(batch)(moved) if isinstance(batch, tuple) else moved
            else:
                self._next = self._to(batch)

            if isinstance(self._next, (list, tuple)):
                self._next = type(self._next)(self._fmt(x) for x in self._next)
            else:
                self._next = self._fmt(self._next)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        out = self._next
        self._preload()
        return out
