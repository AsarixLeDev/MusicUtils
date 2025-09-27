from __future__ import annotations

__all__ = ["setup_matplotlib", "LivePlotDual"]


def setup_matplotlib(backend: str = "auto"):
    """
    Returns (plt, used_backend, ok_interactive).
    Falls back to 'Agg' if interactive backends are unavailable.
    """
    try:
        import matplotlib
    except Exception:
        return None, "", False

    used = ""
    ok_interactive = False

    if backend.lower() == "auto":
        for candidate in ("Qt5Agg", "TkAgg", "WXAgg"):
            try:
                matplotlib.use(candidate, force=True)
                import matplotlib.pyplot as plt  # noqa
                used = candidate
                ok_interactive = True
                break
            except Exception:
                continue
        if not ok_interactive:
            try:
                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt  # noqa
                used = "Agg"
                ok_interactive = False
            except Exception:
                return None, "", False
    else:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt  # noqa
            used = backend
            ok_interactive = used.lower() != "agg"
        except Exception:
            try:
                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt  # noqa
                used = "Agg"
                ok_interactive = False
            except Exception:
                return None, "", False

    import matplotlib.pyplot as plt  # type: ignore
    if ok_interactive:
        plt.ion()
    return plt, used, ok_interactive


class LivePlotDual:
    """
    Two-pane live plot:
      - left: current-epoch train loss (resets each epoch)
      - right: EMA-smoothed global train loss + per-epoch val points
    """

    def __init__(self, plt, smooth: float = 0.10, save_path: str = ""):
        self.plt = plt
        self.smooth = max(0.0, min(1.0, float(smooth)))
        self.save_path = save_path

        # epoch pane
        self.ep_x, self.ep_y = [], []

        # session pane
        self.s_tr_x, self.s_tr_y = [], []
        self.s_va_x, self.s_va_y = [], []

        self.fig, (self.ax_ep, self.ax_s) = self.plt.subplots(1, 2, figsize=(11, 4.5))
        (self.l_ep,) = self.ax_ep.plot([], [], label="train (epoch)")
        self.ax_ep.set_title("Current epoch")
        self.ax_ep.set_xlabel("step in epoch")
        self.ax_ep.set_ylabel("loss")
        self.ax_ep.grid(True, alpha=0.3)

        (self.l_s_tr,) = self.ax_s.plot([], [], label="train (EMA)")
        (self.l_s_va,) = self.ax_s.plot([], [], "o", label="val (per epoch)", markersize=3)
        self.ax_s.set_title("Whole session")
        self.ax_s.set_xlabel("global step / epoch")
        self.ax_s.set_ylabel("loss")
        self.ax_s.grid(True, alpha=0.3)
        self.ax_s.legend()

        self.fig.tight_layout()
        try:
            self.fig.show()
        except Exception:
            pass

    def reset_epoch(self):
        self.ep_x.clear();
        self.ep_y.clear()
        self.l_ep.set_data([], [])
        self._refresh()

    def add_epoch_train(self, step_in_epoch: int, loss: float):
        self.ep_x.append(int(step_in_epoch))
        self.ep_y.append(float(loss))
        self.l_ep.set_data(self.ep_x, self.ep_y)
        self._refresh()

    def add_session_train(self, global_step: int, loss: float):
        y = float(loss)
        if self.s_tr_y:
            y = self.smooth * y + (1.0 - self.smooth) * self.s_tr_y[-1]
        self.s_tr_x.append(int(global_step))
        self.s_tr_y.append(y)
        self.l_s_tr.set_data(self.s_tr_x, self.s_tr_y)
        self._refresh()

    def add_val(self, epoch: int, val_loss: float):
        self.s_va_x.append(int(epoch))
        self.s_va_y.append(float(val_loss))
        self.l_s_va.set_data(self.s_va_x, self.s_va_y)
        self._refresh()

    def _refresh(self):
        try:
            self.ax_ep.relim();
            self.ax_ep.autoscale_view()
            self.ax_s.relim();
            self.ax_s.autoscale_view()
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
            if self.save_path:
                self.fig.savefig(self.save_path, dpi=120, bbox_inches="tight")
        except Exception:
            pass
