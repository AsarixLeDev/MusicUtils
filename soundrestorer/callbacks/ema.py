from soundrestorer.callbacks.callbacks import Callback
from soundrestorer.ema.ema import EMA

import torch


class EMACallback(Callback):
    """Legacy EMA (update on batch_end; swap for val). Prefer EmaUpdateCallback + EmaEvalSwap."""

    def __init__(self, model, decay=0.999):
        self.ema = EMA(model, decay=float(decay))
        self._backup = None

    def on_batch_end(self, trainer=None, **_):
        self.ema.update(trainer.model)

    def on_val_start(self, trainer=None, **_):
        m = trainer.model
        self._backup = {k: v.detach().clone() for k, v in m.state_dict().items()}
        self.ema.apply_to(m)

    def on_val_end(self, trainer=None, **_):
        if self._backup:
            trainer.model.load_state_dict(self._backup)
            self._backup = None

class EmaUpdateCallback(Callback):
    """Keeps EMA weights up-to-date during training."""

    def __init__(self, ema):
        self.ema = ema

    @torch.no_grad()
    def on_batch_end(self, trainer=None, **_):
        self.ema.update(trainer.model)

class EmaEvalSwap(Callback):
    """
    Swap live model weights with EMA for validation, then restore.
    The actual load_state_dict copies are done under `torch.inference_mode()`
    to allow in-place writes on inference-tagged storages.
    """

    def __init__(self, ema_obj, enable: bool = True):
        self.ema = ema_obj
        self.enable = bool(enable)
        self._backup = None
        self._was_training = None

    @torch.no_grad()
    def on_val_start(self, trainer=None, **_):
        if not self.enable or self.ema is None:
            return
        self._was_training = trainer.model.training
        # CPU backup to avoid inference flag issues
        self._backup = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        dev = next(trainer.model.parameters()).device
        ema_sd = {k: v.detach().to(dev) for k, v in self.ema.state_dict().items()}
        with torch.inference_mode():
            trainer.model.load_state_dict(ema_sd, strict=False)
        trainer.model.eval()

    @torch.no_grad()
    def on_val_end(self, trainer=None, **_):
        if self._backup is None:
            return
        dev = next(trainer.model.parameters()).device
        orig_sd = {k: v.to(dev) for k, v in self._backup.items()}
        with torch.inference_mode():
            trainer.model.load_state_dict(orig_sd, strict=False)
        trainer.model.train(self._was_training is True)
        self._backup = None
        self._was_training = None