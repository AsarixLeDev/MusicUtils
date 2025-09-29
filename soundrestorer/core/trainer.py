from __future__ import annotations

import math
import time
from typing import Dict, Any, List, Optional

import torch
from torch import nn
from tqdm import tqdm

from soundrestorer.core.batch_guard import BatchGuard
from soundrestorer.core.prefetch import CUDAPrefetcher
from soundrestorer.core.debug import param_counts, short_device_summary, lr_of, cuda_mem_mb, \
    fmt_compact_comps, fmt_skip_reasons
from soundrestorer.utils.audio import si_sdr_db  # ← use the robust version here


class TrainState:
    def __init__(self):
        self.epoch = 1
        self.global_step = 0
        self.best = None
        self.info: Dict[str, Any] = {}


class Callback:
    def on_train_start(self, **kw): pass
    def on_epoch_start(self, **kw): pass
    def on_batch_start(self, **kw): pass
    def on_batch_end(self, **kw): pass
    def on_val_start(self, **kw): pass
    def on_val_end(self, **kw): pass
    def on_epoch_end(self, **kw): pass
    def on_train_end(self, **kw): pass
    should_stop: bool = False  # optional early stop flag


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
        task,        # implements: step(model, batch) -> (outputs, per_sample_proxy)
        loss_fn,     # ComposedLoss-like: returns (scalar, dict of components)
        optimizer,
        scheduler=None,
        device="cuda",
        amp="float32",  # "bfloat16" | "float16" | "float32"
        grad_accum: int = 1,
        grad_clip: float = 0.0,
        channels_last: bool = True,
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

        # AMP
        self.autocast_dtype = torch.bfloat16 if self.amp == "bfloat16" else (
            torch.float16 if self.amp == "float16" else torch.float32
        )
        self.use_autocast = self.amp in ("bfloat16", "float16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp == "float16"))

        self.model.to(self.device)

    def _zero(self):
        self.opt.zero_grad(set_to_none=True)

    def _step_sched(self):
        if self.sched is None:
            return
        self.sched.step()  # our sched implements .step()

    def fit(self, train_loader, val_loader, epochs: int, ckpt_saver=None):
        debug_cfg = getattr(self, "debug_cfg", {}) or {}
        val_every = int(debug_cfg.get("val_every", 1))  # run validation every N epochs
        val_max_batches = int(debug_cfg.get("val_max_batches", 0))  # 0 = all
        profile_n = int(debug_cfg.get("profile_first_steps", 0))  # 0 = off
        # -------------------- knobs --------------------
        runtime = getattr(self, "runtime", {}) or {}
        guard_cfg = getattr(self, "guard_cfg", {}) or {}
        debug_cfg = getattr(self, "debug_cfg", {}) or {}

        # robust means
        trim_frac = float(getattr(self, "train_trim_frac", 0.10))   # train trimmed mean
        val_trim_frac = float(getattr(self, "val_trim_frac", 0.05)) # val trimmed mean

        # debug cadence
        step_debug = int(debug_cfg.get("step_interval", 0))  # 0 = off
        pbar_every = int(debug_cfg.get("pbar_every", 8))
        show_pair = bool(debug_cfg.get("print_pair_sisdr", False))
        show_mem = bool(debug_cfg.get("print_cuda_mem", False))
        show_comp = bool(debug_cfg.get("print_comp", True))

        # guard enable switch
        guard_enabled = bool(guard_cfg.get("enable", True))

        # -------------------- helpers --------------------
        def _trimmed_mean_tensor(vals: torch.Tensor, trim: float) -> float:
            if vals.numel() == 0:
                return 0.0
            k = int(vals.numel() * trim)
            if k <= 0:
                return float(vals.mean().item())
            v_sorted, _ = torch.sort(vals)
            core = v_sorted[k: v_sorted.numel() - k]
            if core.numel() == 0:
                core = v_sorted
            return float(core.mean().item())

        def _to_cpu_tensor(x) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu", non_blocking=True)
            try:
                return torch.tensor(float(x), dtype=torch.float32)
            except Exception:
                return torch.tensor(0.0, dtype=torch.float32)

        # -------------------- one-time summary --------------------
        try:
            bs = getattr(train_loader, "batch_size", None)
            nw = getattr(train_loader, "num_workers", None)
            pf = getattr(train_loader, "prefetch_factor", None)
            pin = getattr(train_loader, "pin_memory", None)
            per = getattr(train_loader, "persistent_workers", None)
        except Exception:
            bs = nw = pf = pin = per = None

        pc = param_counts(self.model)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * int(epochs)

        compile_desc = runtime.get("compile_desc", "off")
        print("\n===== TRAINING SUMMARY =====")
        print(f"{short_device_summary(self.model, runtime)}")
        print(f"params: {pc['trainable']:,} trainable / {pc['total']:,} total")
        print(f"batch={bs} | workers={nw} | prefetch={pf} | pin={pin} | persistent={per}")
        print(f"epochs={epochs} | steps/epoch={steps_per_epoch} | total_steps={total_steps}")
        print(f"grad_accum={self.grad_accum} | grad_clip={self.grad_clip} | optimizer={type(self.opt).__name__}")
        print(f"initial LR={lr_of(self.opt):.2e} | scheduler={type(self.sched).__name__ if self.sched else 'none'}")
        print(
            f"prefetch={'on' if bool(runtime.get('cuda_prefetch', True)) else 'off'} | "
            f"compile={'on' if compile_desc != 'off' else 'off'} ({compile_desc})"
        )
        print("================================\n")

        # ---------- optional GPU warm-up (once) ----------
        wb = int(runtime.get("warmup_batches", 0) or 0)
        if wb > 0:
            try:
                print(f"[warmup] running {wb} warm-up batches...")
                it_wu = iter(train_loader)
                self.model.eval()
                with torch.inference_mode(), torch.autocast(
                    device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                    dtype=self.autocast_dtype, enabled=self.use_autocast
                ):
                    for _ in tqdm(range(wb)):
                        b = next(it_wu, None)
                        if b is None:
                            break
                        bb = []
                        for x in b:
                            if isinstance(x, torch.Tensor):
                                bb.append(x.to(self.device, non_blocking=True))
                            else:
                                bb.append(x)
                        _ = self.task.step(self.model, bb)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                self.model.train()
                print("[warmup] done.")
            except Exception as e:
                print(f"[warmup] skipped: {e}")

            # clear any cached windows (MR-STFT etc.) built under inference
            try:
                if hasattr(self.task, "_win"):
                    self.task._win = None
            except Exception:
                pass
            try:
                if hasattr(self.loss_fn, "items"):
                    for name, w, fn in self.loss_fn.items:
                        for attr in ("_wins", "_windows"):
                            if hasattr(fn, attr):
                                setattr(fn, attr, {})
            except Exception:
                pass

        # -------------------- BatchGuard --------------------
        guard = BatchGuard(
            hard_clip=float(guard_cfg.get("hard_clip", 0.0)),
            window=int(guard_cfg.get("window", 512)),
            mad_k=float(guard_cfg.get("mad_k", 6.0)),
            snr_floor_db=float(guard_cfg.get("snr_floor_db", -6.0)),
            min_rms_db=float(guard_cfg.get("min_rms_db", -70.0)),
            max_peak=float(guard_cfg.get("max_peak", 1.2)),
        )
        skip_reasons: Dict[str, int] = {}

        st = self.state
        for cb in self.callbacks:
            cb.on_train_start(trainer=self, state=st)

        # prefetchers
        use_prefetch = bool(runtime.get("cuda_prefetch", True)) and self.device.startswith("cuda")
        prefetch_tr = None
        if use_prefetch:
            try:
                prefetch_tr = CUDAPrefetcher(train_loader, device=self.device, channels_last=self.channels_last)
            except Exception as e:
                print(f"[prefetch] disabled for train: {e}")
                prefetch_tr = None

        prefetch_va = None
        if use_prefetch:
            try:
                prefetch_va = CUDAPrefetcher(val_loader, device=self.device, channels_last=self.channels_last)
            except Exception as e:
                print(f"[prefetch] disabled for val: {e}")
                prefetch_va = None

        for epoch in range(st.epoch, epochs + 1):
            st.epoch = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(trainer=self, state=st)

            # -------------------- TRAIN --------------------
            self.model.train()
            t_epoch = time.time()
            used = skipped = 0
            seen_items = 0

            loss_buf_cpu: List[torch.Tensor] = []
            comp_buf_cpu: Dict[str, List[torch.Tensor]] = {}

            ema_loss: Optional[float] = None
            ema_alpha = 0.10

            if prefetch_tr is not None:
                if hasattr(prefetch_tr, "reset"):
                    prefetch_tr.reset()
                get_batch = prefetch_tr.next
            else:
                it = iter(train_loader)
                get_batch = lambda: next(it, None)  # noqa: E731

            from collections import deque
            recent_mad = deque(maxlen=64)
            rescue_after = int(runtime.get("guard_rescue_after", 32))
            rescue_ratio = float(runtime.get("guard_rescue_skip_ratio", 0.5))

            pbar_tr = tqdm(total=len(train_loader), desc=f"train {epoch:03d}", leave=False, dynamic_ncols=True)
            step_in_epoch = 0

            # CUDA Graphs (optional)
            use_graphs = bool(runtime.get("cuda_graph", False))
            graphs_ready = False
            if use_graphs and self.device.startswith("cuda") and (self.grad_accum == 1) and (not self.scaler.is_enabled()):
                try:
                    first = get_batch()
                    if first is not None:
                        from soundrestorer.core.cudagraphs import GraphStep
                        graphstep = GraphStep(self, first)
                        graphs_ready = True
                        batch = get_batch()
                    else:
                        graphs_ready = False
                        batch = get_batch()
                except Exception as e:
                    print(f"[graphs] disabled: {e}")
                    graphs_ready = False
                    batch = get_batch()
            else:
                batch = get_batch()

            while batch is not None:
                if profile_n > 0 and self.device.startswith("cuda"):
                    start_evt = torch.cuda.Event(enable_timing=True);
                    fwd_evt = torch.cuda.Event(enable_timing=True)
                    bwd_evt = torch.cuda.Event(enable_timing=True);
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                for cb in self.callbacks:
                    cb.on_batch_start(trainer=self, state=st, batch=batch)

                # throughput stats
                try:
                    bsz = batch[0].shape[0] if isinstance(batch, (list, tuple)) else 0
                except Exception:
                    bsz = 0

                # forward + loss
                if graphs_ready:
                    outputs, per_sample = graphstep.run(batch)
                    # cheap scalar for pbar
                    s_cpu = _to_cpu_tensor(self.loss_fn(outputs, batch)[0])
                    s_float = float(s_cpu)
                    loss = None
                    comps = {}
                    did_optim_step = False  # GraphStep likely performs stepping; if so, it should also step LR
                else:
                    with torch.autocast(
                        device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                        dtype=self.autocast_dtype, enabled=self.use_autocast
                    ):
                        outputs, per_sample = self.task.step(self.model, batch)
                        loss, comps = self.loss_fn(outputs, batch)
                    s_cpu = loss.detach().to("cpu", non_blocking=True)
                    s_float = float(s_cpu)
                    did_optim_step = False

                # guard
                if guard_enabled:
                    try:
                        skip, why = guard.should_skip(batch=batch, outputs=outputs,
                                                      loss_tensor=(loss.detach() if loss is not None else s_cpu))
                    except TypeError:
                        skip, why = guard.should_skip(batch=batch, outputs=outputs, loss_value=s_float)
                else:
                    skip, why = (False, None)

                if skip:
                    skipped += 1
                    if why:
                        skip_reasons[why] = skip_reasons.get(why, 0) + 1
                        pbar_tr.set_postfix_str(f"skip: {why}")
                    pbar_tr.update(1)
                    batch = get_batch()
                    if why and str(why).startswith("MAD outlier"):
                        recent_mad.append(1)
                    else:
                        recent_mad.append(0)

                    if len(recent_mad) >= rescue_after and (sum(recent_mad) / len(recent_mad)) > rescue_ratio:
                        print("[guard] too many MAD skips -> relaxing thresholds")
                        try:
                            guard.relax()
                        except Exception:
                            pass
                        recent_mad.clear()
                    continue

                if profile_n > 0 and self.device.startswith("cuda"):
                    fwd_evt.record()

                # backward / step
                if (loss is not None) and self.scaler.is_enabled():  # float16 AMP
                    self.scaler.scale(loss / self.grad_accum).backward()
                    if (st.global_step + 1) % self.grad_accum == 0:
                        self.scaler.unscale_(self.opt)
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self._zero()
                        did_optim_step = True
                elif (loss is not None):
                    (loss / self.grad_accum).backward()
                    if (st.global_step + 1) % self.grad_accum == 0:
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.opt.step()
                        self._zero()
                        did_optim_step = True

                if profile_n > 0 and self.device.startswith("cuda"):
                    bwd_evt.record()

                # scheduler & counters
                if did_optim_step:
                    self._step_sched()
                st.global_step += 1
                used += 1
                seen_items += bsz


                # async accumulate
                loss_buf_cpu.append(s_cpu)
                if isinstance(comps, dict):
                    for k, v in comps.items():
                        comp_buf_cpu.setdefault(k, []).append(_to_cpu_tensor(v))

                # throttled live pbar
                if (pbar_every > 0) and (step_in_epoch % pbar_every == 0):
                    if ema_loss is None:
                        ema_loss = s_float
                    else:
                        ema_loss = ema_alpha * s_float + (1 - ema_alpha) * ema_loss
                    post = dict(loss=f"{s_float:.4f}", ema=f"{ema_loss:.4f}", lr=f"{lr_of(self.opt):.2e}")
                    if show_mem:
                        post["mem"] = cuda_mem_mb()
                    pbar_tr.set_postfix(**post)

                # optional per-step debug
                if step_debug > 0 and (st.global_step % step_debug == 0) and show_pair:
                    try:
                        with torch.no_grad():
                            si_noisy = si_sdr_db(batch[0], batch[1]).mean().item()
                            si_rest = si_sdr_db(outputs["yhat"], batch[1]).mean().item()
                        print(f"[S{st.global_step:05d}] loss≈{ema_loss:.3f} | SI {si_noisy:.1f}→{si_rest:.1f}")
                    except Exception:
                        pass

                # notify callbacks (EMA, hard-mining need this)
                for cb in self.callbacks:
                    try:
                        cb.on_batch_end(trainer=self, state=st, batch=batch,
                                        outputs=outputs, per_sample=per_sample)
                    except Exception:
                        pass

                if profile_n > 0 and self.device.startswith("cuda"):
                    end_evt.record();
                    torch.cuda.synchronize()
                    fwd_ms = start_evt.elapsed_time(fwd_evt)
                    bwd_ms = fwd_evt.elapsed_time(bwd_evt)
                    post_ms = bwd_evt.elapsed_time(end_evt)
                    print(f"[prof] fwd {fwd_ms:.1f} ms | bwd {bwd_ms:.1f} ms | post {post_ms:.1f} ms")
                    profile_n -= 1
                pbar_tr.update(1)
                step_in_epoch += 1
                batch = get_batch()

            pbar_tr.close()
            if used == 0 and steps_per_epoch > 0:
                print("[guard] all batches skipped this epoch -> disabling MAD gate temporarily")
                try:
                    guard.enable_mad = False
                except Exception:
                    pass

            # epoch train means
            loss_stack = torch.stack(loss_buf_cpu) if loss_buf_cpu else torch.tensor([], dtype=torch.float32)
            avg_tr = float(loss_stack.mean().item()) if loss_stack.numel() else 0.0
            avg_tr_rob = _trimmed_mean_tensor(loss_stack, trim_frac) if loss_stack.numel() else 0.0

            comps_mean_tr = {}
            for k, lst in sorted(comp_buf_cpu.items()):
                t = torch.stack(lst) if lst else torch.tensor([], dtype=torch.float32)
                comps_mean_tr[k] = float(t.mean().item()) if t.numel() else 0.0

            epoch_time = time.time() - t_epoch
            it_per_s = steps_per_epoch / max(1e-6, epoch_time)
            samp_per_s = seen_items / max(1e-6, epoch_time)

            # -------------------- VALID --------------------
            run_val = (val_every <= 1) or ((st.epoch % val_every) == 0)
            if not run_val:
                print(f"[val] Skipping validation this epoch. val_every={val_every}, st.epoch={st.epoch}")
                # no validation this epoch: propagate a reasonable placeholder
                for cb in self.callbacks:
                    cb.on_val_end(trainer=self, state=st, train_loss=avg_tr_rob, val_loss=avg_tr_rob,
                                  train_comps=comps_mean_tr, val_comps=comps_mean_tr,
                                  train_used=used, train_skipped=skipped, epoch_time=epoch_time)
                for cb in self.callbacks:
                    cb.on_epoch_end(trainer=self, state=st, train_loss=avg_tr_rob, val_loss=avg_tr_rob,
                                    train_comps=comps_mean_tr, val_comps=comps_mean_tr,
                                    train_used=used, train_skipped=skipped, epoch_time=epoch_time)
                continue

            self.model.eval()
            for cb in self.callbacks:
                cb.on_val_start(trainer=self, state=st)

            # --- set up validation iterator/prefetcher for THIS epoch ---
            if prefetch_va is not None:
                # IMPORTANT: reset val prefetcher every epoch
                if hasattr(prefetch_va, "reset"):
                    prefetch_va.reset()
                _val_next = prefetch_va.next
            else:
                val_iter = iter(val_loader)

                def _val_next():
                    return next(val_iter, None)

            try:
                if bool(self.debug_cfg.get("val_no_clean", True)):
                    ds = getattr(val_loader, "dataset", None)
                    if hasattr(ds, "p_clean"):
                        if float(getattr(ds, "p_clean")) != 0.0:
                            ds.p_clean = 0.0
                            print("[valid] forced dataset.p_clean=0.0")
            except Exception as e:
                print(f"[valid] p_clean tweak skipped: {e}")

            loss_v_buf_cpu: List[torch.Tensor] = []
            comp_v_buf_cpu: Dict[str, List[torch.Tensor]] = {}
            dsi_sum = 0.0
            dsi_cnt = 0

            va_len = len(val_loader)
            if val_max_batches > 0:
                va_len = min(va_len, val_max_batches)
            pbar_va = tqdm(total=va_len, desc=f"valid {epoch:03d}", leave=False, dynamic_ncols=True)
            print("Validation...")

            with torch.no_grad(), torch.autocast(
                device_type=("cuda" if self.device.startswith("cuda") else "cpu"),
                dtype=self.autocast_dtype, enabled=self.use_autocast
            ):
                i = 0
                dsi_list = []  # <— NEW
                batch = _val_next()
                while batch is not None and (val_max_batches == 0 or i < val_max_batches):
                    outputs, _ = self.task.step(self.model, batch)
                    loss_v, comps_v = self.loss_fn(outputs, batch)

                    loss_v_buf_cpu.append(_to_cpu_tensor(loss_v))
                    if isinstance(comps_v, dict):
                        for k, v in comps_v.items():
                            comp_v_buf_cpu.setdefault(k, []).append(_to_cpu_tensor(v))

                    # ΔSI probe (first few batches; same device)
                    if dsi_cnt < 5:
                        try:
                            # Prefer tensors prepared by the Task
                            noisy_d = outputs.get("noisy", batch[0]).to(self.device, non_blocking=True)
                            clean_d = outputs.get("clean", batch[1]).to(self.device, non_blocking=True)
                            yhat_d = outputs["yhat"]

                            # Per-sample SI-SDR
                            si_n_vec = si_sdr_db(noisy_d, clean_d, match_length=True)  # (B,)
                            si_r_vec = si_sdr_db(yhat_d, clean_d, match_length=True)  # (B,)

                            # Ignore batches that are basically clean: keep only items with truly noisy inputs
                            noisy_ceiling_db = float(self.debug_cfg.get("sisdr_ignore_if_noisy_gt_db", 35.0))
                            keep = (si_n_vec < noisy_ceiling_db)

                            if keep.any():
                                dsi_batch = (si_r_vec - si_n_vec)[keep]
                                dsi_sum += float(dsi_batch.mean().item());
                                dsi_cnt += 1
                            else:
                                # No genuinely noisy items in this batch → skip contribution
                                pass
                            try:
                                dsi_all = (si_sdr_db(outputs["yhat"], clean_d, match_length=True) -
                                           si_sdr_db(noisy_d, clean_d, match_length=True))
                                if 'keep' in locals():
                                    dsi_all = dsi_all[keep]
                                if dsi_all.numel():
                                    dsi_list.append(dsi_all.detach().to('cpu'))
                            except Exception:
                                pass
                        except Exception:
                            pass
                        if debug_cfg.get("sisdr_probe_print", False):
                            print(f"[ΔSI probe] kept {int(keep.sum())}/{keep.numel()} | "
                                  f"mean SI(noisy,clean)={si_n_vec[keep].mean().item():.2f} dB | "
                                  f"mean SI(yhat,clean)={si_r_vec[keep].mean().item():.2f} dB | "
                                  f"Δ={dsi_batch:+.2f} dB")

                    if (pbar_every > 0) and (i % pbar_every == 0):
                        pbar_va.set_postfix(loss=f"{float(loss_v.detach().to('cpu')):.4f}")
                    pbar_va.update(1)
                    i += 1
                    batch = _val_next()
            pbar_va.close()
            print("Validation done.")

            loss_v_stack = torch.stack(loss_v_buf_cpu) if loss_v_buf_cpu else torch.tensor([], dtype=torch.float32)
            avg_v = float(loss_v_stack.mean().item()) if loss_v_stack.numel() else 0.0
            if not loss_v_buf_cpu:
                print("[valid] No validation batches processed — check prefetch reset or val dataloader.")
            avg_v_rob = _trimmed_mean_tensor(loss_v_stack, val_trim_frac) if loss_v_stack.numel() else 0.0
            if dsi_list:
                dsi_cat = torch.cat(dsi_list)
                dsi_mean = float(dsi_cat.mean())
                dsi_med = float(dsi_cat.median())
                low, high = float(dsi_cat.quantile(0.25)), float(dsi_cat.quantile(0.75))
                print(
                    f"[ΔSI] kept {dsi_cat.numel()} items | mean {dsi_mean:+.2f} dB | median {dsi_med:+.2f} dB | IQR [{low:+.2f},{high:+.2f}]")

            comps_mean_v = {}
            for k, lst in sorted(comp_v_buf_cpu.items()):
                t = torch.stack(lst) if lst else torch.tensor([], dtype=torch.float32)
                comps_mean_v[k] = float(t.mean().item()) if t.numel() else 0.0

            dsi = (dsi_sum / dsi_cnt) if dsi_cnt > 0 else 0.0

            # callbacks
            for cb in self.callbacks:
                cb.on_val_end(trainer=self, state=st, train_loss=avg_tr_rob, val_loss=avg_v_rob,
                              train_comps=comps_mean_tr, val_comps=comps_mean_v,
                              train_used=used, train_skipped=skipped, epoch_time=epoch_time)

            if ckpt_saver:
                ckpt_saver(self.model, self.opt, self.sched, st)

            # display clamp for readability
            _display_clip = {"mrstft": 5.0, "mel_l1": 5.0, "l1_wave": 5.0,
                             "phase_cosine": 2.0, "highband_l1": 5.0,
                             "sisdr_ratio": 2.0, "energy_anchor": 1.0}
            comps_mean_tr_display = {k: min(v, _display_clip.get(k, 10.0)) for k, v in comps_mean_tr.items()}

            comps_str = fmt_compact_comps(comps_mean_tr_display) if show_comp else ""
            reasons_str = fmt_skip_reasons(skip_reasons)

            lines = []
            lines.append("")  # leading blank
            lines.append(
                f"[E{epoch:03d}] train {avg_tr:.3f} (rob {avg_tr_rob:.3f}) | "
                f"val {avg_v_rob:.3f} | ΔSI {dsi:+.2f} dB | used {used} | skip {skipped}"
            )
            if comps_str:
                lines.append(f"      losses: {comps_str}")
            speed_line = f"      speed:  {it_per_s:.2f} it/s, {samp_per_s:.1f} samp/s | lr {lr_of(self.opt):.2e}"
            if show_mem:
                speed_line += f" | mem {cuda_mem_mb()}"
            speed_line += reasons_str
            lines.append(speed_line)
            print("\n".join(lines) + "\n")

            for cb in self.callbacks:
                cb.on_epoch_end(trainer=self, state=st, train_loss=avg_tr_rob, val_loss=avg_v_rob,
                                train_comps=comps_mean_tr, val_comps=comps_mean_v,
                                train_used=used, train_skipped=skipped, epoch_time=epoch_time)

            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                print("[trainer] early stop requested by callback.")
                break

        for cb in self.callbacks:
            cb.on_train_end(trainer=self, state=st)
