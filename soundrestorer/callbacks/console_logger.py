from soundrestorer.callbacks.callbacks import Callback


class ConsoleLogger(Callback):
    def on_val_end(self, state=None, train_loss=None, val_loss=None,
                   train_comps=None, val_comps=None, train_used=None,
                   train_skipped=None, epoch_time=None, trainer=None, **_):
        lr_now = trainer.opt.param_groups[0]['lr'] if trainer else 0.0
        comps_str = ""
        if isinstance(train_comps, dict) and train_comps:
            comps_str = " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(train_comps.items()))
        print(f"[epoch {state.epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f} "
              f"| used {train_used} | skip {train_skipped} | lr {lr_now:.2e} "
              f"| {epoch_time:.1f}s{comps_str}")