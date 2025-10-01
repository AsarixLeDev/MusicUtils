# ---- Legacy callback base (for core.callbacks compatibility) ----
class Callback:
    # new-style hooks (your current trainer uses these)
    def on_fit_begin(self, trainer): pass
    def on_fit_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch: int): pass
    def on_epoch_end(self, trainer, epoch: int): pass
    def on_train_batch_end(self, trainer, batch_idx: int, batch, outputs): pass

    # legacy hooks (used by soundrestorer.core.callbacks.*)
    def on_epoch_start(self, trainer=None, state=None, **_): pass
    def on_batch_start(self, trainer=None, state=None, batch=None, **_): pass
    def on_batch_end(self, trainer=None, state=None, batch=None, outputs=None, **_): pass
    def on_val_start(self, trainer=None, state=None, **_): pass
    def on_val_end(self, trainer=None, state=None, train_loss=None, val_loss=None,
                   train_comps=None, val_comps=None, train_used=None, train_skipped=None,
                   epoch_time=None, **_): pass

    should_stop: bool = False  # some callbacks may set this
