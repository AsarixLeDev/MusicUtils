from soundrestorer.callbacks.callbacks import Callback


class EarlyStoppingCallback(Callback):
    def __init__(self, patience=10, min_delta=0.0, monitor="val_loss", mode="min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode.lower()
        self.best = None
        self.count = 0
        self.should_stop = False

    def _improved(self, current):
        if self.best is None:
            return True
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def on_val_end(self, train_loss=None, val_loss=None, **_):
        current = val_loss if self.monitor == "val_loss" else train_loss
        if self._improved(current):
            self.best = current
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
                print(f"[early-stop] no improvement for {self.patience} evals. Stopping.")