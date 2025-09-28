from __future__ import annotations

import math
import time
import torch
from typing import Dict, Any, List, Optional

from torch import nn
from tqdm import tqdm

from soundrestorer.core.batch_guard import BatchGuard
from soundrestorer.core.prefetch import CUDAPrefetcher
from soundrestorer.core.debug import param_counts, si_sdr_db, short_device_summary, lr_of, cuda_mem_mb


class TrainState:
    def __init__(self):
        self.epoch = 1
        self.global_step = 0
        self.best = None
        self.info: Dict[str, Any] = {}


class Callback:
    def on_train_start(self, **kw): pass

    def on_epoch_start(self, **kw): pass

    def on_batch_start(self, **kw): pass  # <--- NEW

    def on_batch_end(self, **kw): pass

    def on_val_start(self, **kw): pass

    def on_val_end(self, **kw): pass

    def on_epoch_end(self, **kw): pass

    def on_train_end(self, **kw): pass

    # optional flag for early stopping:
    should_stop: bool = False


class WarmupCosine:
    def __init__(self, optimizer, total_steps: int, warmup: int = 0, min_factor: float = 0.1):
        self.opt = optimizer
        self.tot = max(1, total_steps)
        self.warm = max(0, warmup)
        self.minf = float(min_factor)
        self._step = 0
        self._base = [g['lr'] for g in self.opt.param_groups]

    def step(self):
        self._step += 1
        if self._step <= self.warm:
            s = self._step / max(1, self.warm)
        else:
            prog = (self._step - self.warm) / max(1, self.tot - self.warm)
            s = self.minf + (1 - self.minf) * 0.5 * (1 + math.cos(math.pi * prog))
        for pg, b in zip(self.opt.param_groups, self._base):
            pg['lr'] = b * s


class Trainer:
    """
    Generic trainer that delegates the *step* to a Task object.
    The Task is responsible for turning a batch into model inputs/outputs.
    Losses are computed by ComposedLoss (already assembled).
    """

    def __init__(
            self,
            model: nn.Module,
            task,  # implements: step(model, batch) -> (outputs, per_sample_proxy)
            loss_fn,  # ComposedLoss-like: returns (scalar, dict of components)
            optimizer,
            scheduler=None,
            device="cuda",
            amp="float32",  # "bfloat16" | "float16" | "float32"
            grad_accum: int = 1,
            grad_clip: float = 0.0,
            channels_last: bool = True,
            compile_model: bool = False,
            callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        if channels_last and device.startswith("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
        self.task = task
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.sched = scheduler
        self.device = device
        self.runtime = getattr(self, "runtime", {})  # or pass it in the constructor
        self.amp = str(amp).lower()
        self.grad_accum = max(1, int(grad_accum))
        self.grad_clip = float(grad_clip)
        self.channels_last = bool(channels_last)
        self.callbacks = callbacks or []
        self.state = TrainState()

        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[compile] torch.compile active")
            except Exception as e:
                print(f"[compile] disabled: {e}")

        # AMP
        self.autocast_dtype = torch.bfloat16 if self.amp == "bfloat16" else (
            torch.float16 if self.amp == "float16" else torch.float32
        )
        self.use_autocast = self.amp in ("bfloat16", "float16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp == "float16"))

        if self.channels_last and (device.startswith("cuda")):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model.to(self.device)

    def _zero(self):
        self.opt.zero_grad(set_to_none=True)

    def _step_sched(self):
        if self.sched is None: return
        # support LambdaLR or our WarmupCosine
        if hasattr(self.sched, "step"): self.sched.step()

    def fit(self, train_loader, val_loader, epochs: int, ckpt_saver=None):
        # ---- knobs (overridable from outside) ----
        clip_thr = float(getattr(self, "train_loss_clip", 0.0))  # 0 = disable (BatchGuard handles spikes)
        trim_frac = float(getattr(self, "train_trim_frac", 0.10))  # robust epoch mean
        runtime = getattr(self, "runtime", {}) or {}
        guard_cfg = getattr(self, "guard_cfg", {}) or {}
        debug_cfg = getattr(self, "debug_cfg", {}) or {}
        step_debug = int(debug_cfg.get("step_interval", 100))  # prints every N steps
        show_pair = bool(debug_cfg.get("print_pair_sisdr", True))  # include SI(noisy/clean,yhat/clean)
        show_mem = bool(debug_cfg.get("print_cuda_mem", True))  # mem in postfix
        show_comp = bool(debug_cfg.get("print_comp", True))  # loss components at epoch end

        # ---- helper ----
        def _trimmed_mean(vals, trim):
            if not vals: return 0.0
            vs = sorted(vals);
            k = int(len(vs) * trim)
            core = vs[k: len(vs) - k] if len(vs) - 2 * k > 0 else vs
            return sum(core) / max(1, len(core))

        # ---- training summary (once) ----
        try:
            bs = getattr(train_loader, "batch_size", None)
            nw = getattr(train_loader, "num_workers", None)
            pf = getattr(train_loader, "prefetch_factor", None)
            pin = getattr(train_loader, "pin_memory", None)
            pers = getattr(train_loader, "persistent_workers", None)
        except Exception:
            bs = nw = pf = pin = pers = None

        pc = param_counts(self.model)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * int(epochs)
        print("\n===== TRAINING SUMMARY =====")
        print(f"{short_device_summary(self.model, runtime)}")
        print(f"params: {pc['trainable']:,} trainable / {pc['total']:,} total")
        print(f"batch={bs} | workers={nw} | prefetch={pf} | pin={pin} | persistent={pers}")
        print(f"epochs={epochs} | steps/epoch={steps_per_epoch} | total_steps={total_steps}")
        print(f"grad_accum={self.grad_accum} | grad_clip={self.grad_clip} | optimizer={type(self.opt).__name__}")
        print(f"initial LR={lr_of(self.opt):.2e} | scheduler={type(self.sched).__name__ if self.sched else 'none'}")
        print(
            f"prefetch={'on' if (bool(runtime.get('cuda_prefetch', True)) and self.device.startswith('cuda')) else 'off'} "
            f"| guard={guard_cfg if guard_cfg else 'off'}")
        print("================================\n")

        # ---- batch guard ----
        from soundrestorer.core.batch_guard import BatchGuard
        guard = BatchGuard(
            hard_clip=float(guard_cfg.get("hard_clip", 0.0)),  # static clip off by default
            window=int(guard_cfg.get("window", 512)),
            mad_k=float(guard_cfg.get("mad_k", 6.0)),
            snr_floor_db=float(guard_cfg.get("snr_floor_db", -6.0)),
            min_rms_db=float(guard_cfg.get("min_rms_db", -70.0)),
            max_peak=float(guard_cfg.get("max_peak", 1.2)),
        )
        skip_reasons: Dict[str, int] = {}

        st = self.state
        for cb in self.callbacks: cb.on_train_start(trainer=self, state=st)

        for epoch in range(st.epoch, epochs + 1):
            st.epoch = epoch
            for cb in self.callbacks: cb.on_epoch_start(trainer=self, state=st)

            # ---------- TRAIN ----------
            self.model.train()
            t_epoch = time.time()
            used = skipped = 0
            sum_loss = 0.0
            train_losses = []
            comp_sums_tr, comp_count_tr = {}, 0
            seen_items = 0

            # prefetch / iterator
            use_prefetch = bool(runtime.get("cuda_prefetch", True)) and self.device.startswith("cuda")
            if use_prefetch:
                try:
                    from soundrestorer.core.prefetch import CUDAPrefetcher
                    prefetch = CUDAPrefetcher(train_loader, device=self.device, channels_last=self.channels_last)
                    get_batch = prefetch.next
                except Exception as e:
                    print(f"[prefetch] disabled: {e}")
                    it = iter(train_loader);
                    get_batch = lambda: next(it, None)
            else:
                it = iter(train_loader);
                get_batch = lambda: next(it, None)

            pbar_tr = tqdm(total=len(train_loader), desc=f"train {epoch:03d}", leave=False, dynamic_ncols=True)
            step_in_epoch = 0
            t_loop = time.time()

            batch = get_batch()
            while batch is not None:
                for cb in self.callbacks: cb.on_batch_start(trainer=self, state=st, batch=batch)

                # (B,...,T) count for throughput (best-effort)
                try:
                    bsz = batch[0].shape[0] if isinstance(batch, (list, tuple)) else 0
                except Exception:
                    bsz = 0

                # forward + loss
                with torch.autocast(device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                                    dtype=self.autocast_dtype, enabled=self.use_autocast):
                    outputs, per_sample = self.task.step(self.model, batch)
                    loss, comps = self.loss_fn(outputs, batch)

                s = float(loss.detach().item())
                # guard (NaN/Inf/RMS/peak/SNR/MAD) + optional static clip
                skip, why = guard.should_skip(batch=batch, outputs=outputs, loss_value=s)
                if skip or (clip_thr > 0 and s > clip_thr) or (not math.isfinite(s)):
                    skipped += 1
                    if why:
                        skip_reasons[why] = skip_reasons.get(why, 0) + 1
                        pbar_tr.set_postfix_str(f"skip: {why}")
                    pbar_tr.update(1)
                    batch = get_batch()
                    continue

                if step_debug > 0 and (st.global_step % step_debug == 0):
                    try:
                        R = outputs.get("R");
                        I = outputs.get("I")
                        if isinstance(R, torch.Tensor) and isinstance(I, torch.Tensor):
                            mag = torch.sqrt(R.float() ** 2 + I.float() ** 2 + 1e-8)
                            m_mean = float(mag.mean())
                            m_min = float(mag.min())
                            m_max = float(mag.max())
                            m_str = f" | |M| mean={m_mean:.3f} min={m_min:.3f} max={m_max:.3f}"
                        else:
                            m_str = ""
                        if show_pair:
                            with torch.no_grad():
                                noisy, clean = batch[0], batch[1]
                                si_noisy = si_sdr_db(noisy, clean).mean().item()
                                si_rest = si_sdr_db(outputs['yhat'], clean).mean().item()
                        else:
                            si_noisy = si_rest = float('nan')
                        mem = f" | mem {cuda_mem_mb()}" if show_mem else ""
                        print(f"[dbg] ep{epoch} step{st.global_step} | loss {s:.4f} | "
                              f"SI noisy {si_noisy:.2f} dB -> rest {si_rest:.2f} dB{m_str}{mem}")
                    except Exception:
                        pass

                # backward / step
                if self.scaler.is_enabled():
                    self.scaler.scale(loss / self.grad_accum).backward()
                    if (st.global_step + 1) % self.grad_accum == 0:
                        self.scaler.unscale_(self.opt)
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.opt);
                        self.scaler.update();
                        self._zero()
                else:
                    (loss / self.grad_accum).backward()
                    if (st.global_step + 1) % self.grad_accum == 0:
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.opt.step();
                        self._zero()

                # scheduler + book-keeping
                self._step_sched()
                st.global_step += 1
                used += 1
                sum_loss += s
                train_losses.append(s)
                seen_items += bsz
                if isinstance(comps, dict):
                    for k, v in comps.items():
                        comp_sums_tr[k] = comp_sums_tr.get(k, 0.0) + float(v)
                    comp_count_tr += 1

                # per-batch callbacks
                for cb in self.callbacks:
                    cb.on_batch_end(trainer=self, state=st, loss=s, comps=comps, outputs=outputs,
                                    per_sample=per_sample, batch=batch)

                # step debug print (SI metrics etc.)
                step_in_epoch += 1
                if step_debug > 0 and (st.global_step % step_debug == 0):
                    try:
                        if show_pair:
                            with torch.no_grad():
                                noisy, clean = batch[0], batch[1]
                                si_noisy = si_sdr_db(noisy, clean).mean().item()
                                si_rest = si_sdr_db(outputs["yhat"], clean).mean().item()
                        else:
                            si_noisy = si_rest = float("nan")
                        mem = f" | mem {cuda_mem_mb()}" if show_mem else ""
                        print(f"[dbg] ep{epoch} step{st.global_step} "
                              f"| loss {s:.4f} | SI noisy {si_noisy:.2f} dB -> rest {si_rest:.2f} dB{mem}")
                    except Exception:
                        pass

                # progress bar
                lr_now = lr_of(self.opt)
                avg_now = (sum_loss / max(1, used))
                rob_now = _trimmed_mean(train_losses, trim_frac)
                postfix = dict(loss=f"{s:.4f}", avg=f"{avg_now:.4f}", rob=f"{rob_now:.4f}", lr=f"{lr_now:.2e}")
                if show_mem: postfix["mem"] = cuda_mem_mb()
                pbar_tr.set_postfix(**postfix)
                pbar_tr.update(1)
                batch = get_batch()

            pbar_tr.close()
            avg_tr = (sum_loss / max(1, used))
            avg_tr_rob = _trimmed_mean(train_losses, trim_frac)
            comps_mean_tr = {k: (comp_sums_tr[k] / max(1, comp_count_tr)) for k in sorted(comp_sums_tr.keys())}
            epoch_time = time.time() - t_epoch
            it_per_s = len(train_loader) / max(1e-6, epoch_time)
            samp_per_s = seen_items / max(1e-6, epoch_time)

            # ---------- VALID ----------
            self.model.eval()
            tot_v = 0.0;
            used_v = 0
            comp_sums_v, comp_count_v = {}, 0
            for cb in self.callbacks: cb.on_val_start(trainer=self, state=st)

            pbar_va = tqdm(total=len(val_loader), desc=f"valid {epoch:03d}", leave=False, dynamic_ncols=True)
            with torch.no_grad(), torch.autocast(device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                                                 dtype=self.autocast_dtype, enabled=self.use_autocast):
                for batch in val_loader:
                    outputs, _ = self.task.step(self.model, batch)
                    loss_v, comps_v = self.loss_fn(outputs, batch)
                    sv = float(loss_v.detach().item())
                    if not math.isfinite(sv):
                        pbar_va.update(1);
                        continue
                    tot_v += sv;
                    used_v += 1
                    if isinstance(comps_v, dict):
                        for k, v in comps_v.items():
                            comp_sums_v[k] = comp_sums_v.get(k, 0.0) + float(v)
                        comp_count_v += 1
                    pbar_va.set_postfix(loss=f"{sv:.4f}")
                    pbar_va.update(1)
            pbar_va.close()

            avg_v = tot_v / max(1, used_v)
            comps_mean_v = {k: (comp_sums_v[k] / max(1, comp_count_v)) for k in sorted(comp_sums_v.keys())}

            # callbacks
            for cb in self.callbacks:
                cb.on_val_end(trainer=self, state=st, train_loss=avg_tr_rob, val_loss=avg_v,
                              train_comps=comps_mean_tr, val_comps=comps_mean_v,
                              train_used=used, train_skipped=skipped, epoch_time=epoch_time)

            if ckpt_saver: ckpt_saver(self.model, self.opt, self.sched, st)

            # epoch summary
            lr_now = lr_of(self.opt)
            comps_str = (" " + " ".join(f"{k}={v:.4f}" for k, v in comps_mean_tr.items())) if show_comp else ""
            # show top skip reasons
            if skip_reasons:
                top_reasons = sorted(skip_reasons.items(), key=lambda kv: kv[1], reverse=True)[:4]
                reasons_str = " | skips: " + ", ".join(f"{k} x{v}" for k, v in top_reasons)
            else:
                reasons_str = ""
            print(f"[epoch {epoch:03d}] train {avg_tr:.4f} (rob {avg_tr_rob:.4f}) | val {avg_v:.4f} "
                  f"| used {used} | skip {skipped} | lr {lr_now:.2e} | {epoch_time:.1f}s "
                  f"| it/s {it_per_s:.2f} | samp/s {samp_per_s:.1f}{reasons_str}{comps_str}")

            for cb in self.callbacks:
                cb.on_epoch_end(trainer=self, state=st, train_loss=avg_tr_rob, val_loss=avg_v,
                                train_comps=comps_mean_tr, val_comps=comps_mean_v,
                                train_used=used, train_skipped=skipped, epoch_time=epoch_time)

            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                print("[trainer] early stop requested by callback.")
                break

        for cb in self.callbacks: cb.on_train_end(trainer=self, state=st)

