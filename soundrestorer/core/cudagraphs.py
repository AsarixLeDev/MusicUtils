# soundrestorer/core/cudagraphs.py
from __future__ import annotations
import torch

class GraphStep:
    """
    Capture 1 full train step (forward->loss->backward->optim->zero) and replay it.
    Requirements:
      - static shapes (batch, STFT dims)
      - grad_accum == 1
      - no GradScaler (bf16 AMP is fine)
    """
    def __init__(self, trainer, first_batch):
        self.tr = trainer
        self.dev = trainer.device
        self.graph = None
        self.static_batch = None
        self.static_out = {}
        self._build(first_batch)

    def _clone_like(self, t):
        return t.clone().detach().to(self.dev).requires_grad_(False)

    def _materialize_static(self, batch):
        # Expect tuple/list; clone tensors into persistent/static buffers
        sb = []
        for x in batch:
            if isinstance(x, torch.Tensor):
                sb.append(self._clone_like(x))
            else:
                sb.append(x)  # meta dicts stay on CPU
        return type(batch)(sb) if isinstance(batch, tuple) else sb

    def _copy_into_static(self, batch):
        for s, x in zip(self.static_batch, batch):
            if isinstance(s, torch.Tensor) and isinstance(x, torch.Tensor):
                s.copy_(x, non_blocking=True)

    def _build(self, first_batch):
        # warmup 1 eager step to allocate
        self.tr.model.train()
        with torch.autocast(device_type=("cuda" if self.dev.startswith("cuda") else "cpu"),
                            dtype=self.tr.autocast_dtype, enabled=self.tr.use_autocast):
            out, _ = self.tr.task.step(self.tr.model, first_batch)
            loss, _ = self.tr.loss_fn(out, first_batch)
        loss.backward()
        self.tr.opt.step(); self.tr._zero()
        torch.cuda.synchronize()

        # static buffers
        self.static_batch = self._materialize_static(first_batch)

        self.graph = torch.cuda.CUDAGraph()
        # capture
        self.tr.model.train()
        self.tr.opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            out_g, _ = self.tr.task.step(self.tr.model, self.static_batch)
            loss_g, _ = self.tr.loss_fn(out_g, self.static_batch)
            loss_g.backward()
            self.tr.opt.step()
            self.tr.opt.zero_grad(set_to_none=True)
            # keep a ref to outputs we want to inspect (yhat at least)
            self.static_out["yhat"] = out_g.get("yhat", None)

    def run(self, batch):
        # copy new data into static input buffers and replay
        self._copy_into_static(batch)
        self.graph.replay()
        return self.static_out, None  # no per-sample proxy in graph mode
