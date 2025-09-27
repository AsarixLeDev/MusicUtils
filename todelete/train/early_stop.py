from __future__ import annotations

__all__ = ["EarlyStopper"]


class EarlyStopper:
    """
    Patience-based early stopping on a scalar metric (e.g. validation loss).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.num_bad = 0

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        """
        Returns (should_stop, is_best_now)
        """
        is_best = False
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
            is_best = True
        else:
            self.num_bad += 1
        stop = self.num_bad >= self.patience
        return stop, is_best
