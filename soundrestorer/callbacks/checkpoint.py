import os
import torch

from soundrestorer.callbacks.callbacks import Callback


class CheckpointCallback(Callback):
    def __init__(self, out_dir: str, every: int = 1):
        self.out = out_dir
        self.every = int(every)
        os.makedirs(self.out, exist_ok=True)

    def on_epoch_end(self, trainer=None, state=None, **_):
        if (state.epoch % self.every) != 0:
            return
        p = os.path.join(self.out, f"epoch_{state.epoch:03d}.pt")
        torch.save({
            "model": trainer.model.state_dict(),
            "opt": trainer.opt.state_dict(),
            "sched": getattr(trainer.sched, "state_dict", lambda: {})(),
            "epoch": state.epoch, "global_step": state.global_step
        }, p)
        print(f"[ckpt] saved {p}")