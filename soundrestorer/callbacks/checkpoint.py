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

class BestCheckpointCallback(Callback):
    def __init__(self, out_dir: str, k: int = 3, monitor: str = "val_loss", mode: str = "min"):
        self.out = out_dir
        os.makedirs(self.out, exist_ok=True)
        self.k = int(k)
        self.monitor = monitor
        self.mode = mode.lower()
        self._best = []  # list of (score, path)

    def _better(self, a, b):
        return (a < b) if self.mode == "min" else (a > b)

    def on_val_end(self, trainer=None, state=None, train_loss=None, val_loss=None, **_):
        score = val_loss if self.monitor == "val_loss" else train_loss
        p = os.path.join(self.out, f"best_ep{state.epoch:03d}.pt")
        torch.save({
            "model": trainer.model.state_dict(),
            "opt": trainer.opt.state_dict(),
            "sched": getattr(trainer.sched, "state_dict", lambda: {})(),
            "epoch": state.epoch, "global_step": state.global_step,
            "score": float(score),
        }, p)
        self._best.append((float(score), p))
        self._best.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self._best) > self.k:
            _, worst_p = self._best.pop()
            try:
                os.remove(worst_p)
            except Exception:
                pass
        if self._best:
            best_p = self._best[0][1]
            link = os.path.join(self.out, "best.pt")
            try:
                if os.path.exists(link):
                    os.remove(link)
            except Exception:
                pass
            try:
                import shutil
                shutil.copyfile(best_p, link)
            except Exception:
                pass