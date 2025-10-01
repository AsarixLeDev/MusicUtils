# soundrestorer/core/prefetch.py
from __future__ import annotations

import torch


class CUDAPrefetcher:
    """
    Overlaps H2D copies with compute.
    - Keeps a handle to the source DataLoader (self.src) so we can reset per epoch
    - Moves tensors to device non_blocking=True
    - Leaves meta dicts on CPU
    - Optionally sets channels_last on 4D tensors (B,2,F,T)
    """

    def __init__(self, loader, device: str = "cuda", channels_last: bool = True):
        self.src = loader  # <— keep the DataLoader
        self.device = device
        self.channels_last = bool(channels_last)
        self.stream = torch.cuda.Stream() if device.startswith("cuda") else None
        self.loader = None
        self._next = None
        self.reset()  # <— initialize iterator + first preload

    def reset(self, loader=None):
        """
        Reset the internal iterator (call once per epoch).
        Optionally pass a new DataLoader to switch sources.
        """
        if loader is not None:
            self.src = loader
        self.loader = iter(self.src)
        self._next = None
        self._preload()

    def _to(self, t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.to(self.device, non_blocking=True)
        return t  # meta, dicts, etc.

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
            # CPU-only path: just keep the batch as-is
            self._next = batch
            return

        with torch.cuda.stream(self.stream):
            if isinstance(batch, (list, tuple)):
                moved = []
                for x in batch:
                    if isinstance(x, dict):
                        moved.append(x)  # meta stays on CPU
                    elif isinstance(x, (list, tuple)):
                        moved.append(type(x)(self._to(xx) for xx in x))
                    else:
                        moved.append(self._to(x))
                self._next = type(batch)(moved) if isinstance(batch, tuple) else moved
            else:
                self._next = self._to(batch)

            # format after copy
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
