from soundrestorer.callbacks.callbacks import Callback


class CurriculumCallback(Callback):
    def __init__(self, loss_fn, task, dataset=None,
                 snr_stages=None, sisdr=None, mask_limit=None,
                 mask_variant=None,
                 mask_reg=None,  # <— NEW: schedule weight for mask_unity_reg
                 sisdr_weight=None):  # <— NEW: schedule sisdr_ratio weight
        self.loss_fn = loss_fn
        self.task = task
        self.dataset = dataset
        self.snr_stages = snr_stages or []
        self.sisdr = sisdr or {}
        self.mask_limit = mask_limit or {}
        self.mask_variant = mask_variant or []  # list of {until: int, variant: str}
        self.mask_reg = mask_reg or {}
        self.sisdr_weight = sisdr_weight or {}

    def _snr_for_epoch(self, epoch):
        # pick first stage whose 'until' >= epoch
        take = None
        for st in self.snr_stages:
            if epoch <= int(st.get("until", 10 ** 9)):
                take = st
                break
        if take is None and self.snr_stages:
            take = self.snr_stages[-1]
        return take

    def on_epoch_start(self, state=None, **_):
        e = state.epoch

        # SNR window
        st = self._snr_for_epoch(e)
        if st:
            if self.dataset is not None:
                for k in ("snr_min", "snr_max", "use_ext_noise_p"):
                    if hasattr(self.dataset, k) and (k in st):
                        setattr(self.dataset, k, st[k])
            print(f"[curriculum] epoch {e}: SNR [{st.get('snr_min', '?')},{st.get('snr_max', '?')}]")

        # SISDR target schedule
        if self.sisdr:
            s0 = float(self.sisdr.get("start_db", 0.0))
            s1 = float(self.sisdr.get("end_db", 12.0))
            e1 = int(self.sisdr.get("end_epoch", 20))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            target = (1.0 - t) * s0 + t * s1
            ok = getattr(self.loss_fn, "set_attr", lambda *a, **k: False)("sisdr_ratio", "min_db", target)
            if ok:
                print(f"[curriculum] epoch {e}: SISDR min_db -> {target:.2f} dB")

        # (optional) bump sisdr_ratio weight slightly after warm-up
        if hasattr(self.loss_fn, "items") and e >= 3:
            for i, (name, w, fn) in enumerate(self.loss_fn.items):
                if name == "sisdr_ratio" and w < 0.40:
                    self.loss_fn.items[i] = (name, 0.40, fn)

        # mask_limit schedule on task
        if self.mask_limit and hasattr(self.task, "mask_limit"):
            m0 = float(self.mask_limit.get("start", 1.5))
            m1 = float(self.mask_limit.get("end", 2.5))
            e1 = int(self.mask_limit.get("end_epoch", 20))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            target = (1.0 - t) * m0 + t * m1
            self.task.mask_limit = target
            print(f"[curriculum] epoch {e}: mask_limit -> {target:.2f}")

        if self.mask_reg:
            w0 = float(self.mask_reg.get("start_w", 0.05))
            w1 = float(self.mask_reg.get("end_w", 0.00))
            e1 = int(self.mask_reg.get("end_epoch", 4))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            w = (1.0 - t) * w0 + t * w1
            if hasattr(self.loss_fn, "items"):
                updated = False
                for i, (name, w_old, fn) in enumerate(self.loss_fn.items):
                    if name == "mask_unity_reg":
                        self.loss_fn.items[i] = (name, w, fn)
                        updated = True
                        break
                if updated:
                    print(f"[curriculum] epoch {e}: mask_unity_reg weight -> {w:.3f}")

            # ramp 'sisdr_ratio' weight up later
        if self.sisdr_weight:
            w0 = float(self.sisdr_weight.get("start_w", 0.10))
            w1 = float(self.sisdr_weight.get("end_w", 0.30))
            e1 = int(self.sisdr_weight.get("end_epoch", 8))
            t = min(1.0, max(0.0, (e - 1) / max(1, e1 - 1)))
            w = (1.0 - t) * w0 + t * w1
            if hasattr(self.loss_fn, "items"):
                for i, (name, w_old, fn) in enumerate(self.loss_fn.items):
                    if name == "sisdr_ratio":
                        self.loss_fn.items[i] = (name, w, fn)
                        print(f"[curriculum] epoch {e}: sisdr_ratio weight -> {w:.3f}")
                        break

        # mask_variant stage (delta1 -> plain)
        if self.mask_variant and hasattr(self.task, "mask_variant"):
            for stg in self.mask_variant:
                until = int(stg.get("until", 10 ** 9))
                if e <= until:
                    v = str(stg.get("variant", "")).lower()
                    if v:
                        self.task.mask_variant = v
                        print(f"[curriculum] epoch {e}: mask_variant -> {v}")
                    break