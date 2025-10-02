#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determinism sanity for ProcNoiseAugmentCallback.

Two modes:
  1) synthetic  — build (B,T) clean tone, apply callback deterministically, verify reproducibility.
  2) loader     — run a real DataLoader + callback, with options to fix dataset index, pin the
                  clean segment, and avoid peak re-scaling to ensure identical residuals.

Also saves WAV triads if --save-wavs is given.
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch

# --- project imports ---
from scripts.train import _load_yaml
from soundrestorer.data.builder import build_loaders
from soundrestorer.core.trainer import TrainState
from soundrestorer.callbacks.callbacks import Callback  # base class
from soundrestorer.callbacks.proc_noise_augment import ProcNoiseAugmentCallback
from soundrestorer.callbacks.utils import save_wav  # expects (C,T) float32 in [-1,1]


def _bt(x: torch.Tensor) -> torch.Tensor:
    """(B,C,T)->(B,T) mono; (B,T)->(B,T); (T,)->(1,T)."""
    if x.dim() == 3:
        return x.mean(dim=1)
    if x.dim() == 2:
        return x
    if x.dim() == 1:
        return x.unsqueeze(0)
    raise RuntimeError(f"Unexpected shape {tuple(x.shape)}")


def _ls_residual_unit_rms(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    LS residual after removing clean projection, then unit-RMS normalize.
    Operates on (B,T) tensors.
    """
    n = min(noisy.shape[-1], clean.shape[-1])
    n_bt = noisy[..., :n] - noisy[..., :n].mean(dim=-1, keepdim=True)
    c_bt = clean[..., :n] - clean[..., :n].mean(dim=-1, keepdim=True)

    num = (n_bt * c_bt).sum(dim=-1, keepdim=True)
    den = c_bt.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    alpha = num / den
    resid = n_bt - alpha * c_bt  # (B,T)

    rms = torch.sqrt((resid ** 2).mean(dim=-1, keepdim=True)).clamp_min(1e-12)
    resid = resid / rms
    return resid


def _hash_resid(resid_bt: torch.Tensor) -> str:
    """Hash concatenated (B,T) residual after unit-RMS step."""
    r = resid_bt.detach().to(torch.float32).cpu().contiguous()
    return hashlib.sha1(r.numpy().tobytes()).hexdigest()


def _snr_db(clean_bt: torch.Tensor, noisy_bt: torch.Tensor) -> float:
    n = min(clean_bt.shape[-1], noisy_bt.shape[-1])
    c = clean_bt[..., :n]
    y = noisy_bt[..., :n]
    num = (c ** 2).sum(dim=-1).clamp_min(1e-12)
    den = ((y - c) ** 2).sum(dim=-1).clamp_min(1e-12)
    return float((10.0 * torch.log10(num / den)).mean().item())


def _save_triad(out_dir: Path, base: str, clean_bt: torch.Tensor, noisy_bt: torch.Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    # save as mono (1,T)
    c = clean_bt[0:1].detach().cpu()
    n = noisy_bt[0:1].detach().cpu()
    r = (n - c)
    r_ls = _ls_residual_unit_rms(n, c)

    save_wav(out_dir / f"{base}_clean.wav", c, sr=48000)
    save_wav(out_dir / f"{base}_noisy.wav", n, sr=48000)
    save_wav(out_dir / f"{base}_resid.wav", r, sr=48000)
    save_wav(out_dir / f"{base}_resid_ls.wav", r_ls[0:1].cpu(), sr=48000)


# ------------------------------------------------------------------------------------
# MODES
# ------------------------------------------------------------------------------------

def run_synthetic(args):
    print("\n" + "=" * 80 + "\n1) SYNTHETIC MODE\n" + "=" * 80)

    B, T, sr = 1, int(args.seconds * args.sr), args.sr
    t = torch.linspace(0, args.seconds, T, dtype=torch.float32)
    # simple clean tone
    clean_bt = torch.sin(2 * np.pi * 440.0 * t).unsqueeze(0)  # (1,T)
    noisy_bt = clean_bt.clone()

    cb = ProcNoiseAugmentCallback(sr=sr, prob=args.prob,
                                  snr_min=args.snr_min, snr_max=args.snr_max,
                                  fixed_seed=args.fixed_seed,
                                  fixed_per_epoch=args.fixed_per_epoch,
                                  train_only=False, track_stats=True, out_peak=1.0 if args.no_peak else 0.98)

    state = TrainState()
    state.epoch = 1
    cb.on_epoch_start(state=state)

    # call twice
    for k in range(2):
        batch = {"noisy": noisy_bt.clone(), "clean": clean_bt.clone()}
        cb.on_batch_start(trainer=None, state=state, batch=batch)
        n2 = _bt(batch["noisy"]); c2 = _bt(batch["clean"])
        print(f"[proc-noise] pre-mix SNR ~ +{_snr_db(c2, c2):.2f} dB")
        print(f"[proc-noise] post-mix SNR ~ +{_snr_db(c2, n2):.2f} dB")

    # hash check
    h1 = _hash_resid(_ls_residual_unit_rms(_bt(noisy_bt), _bt(clean_bt)))
    h2 = _hash_resid(_ls_residual_unit_rms(_bt(batch["noisy"]), _bt(batch["clean"])))
    print("Hashes:")
    print(f"  before: {h1}")
    print(f"  after : {h2}")
    print("\nDeterminism verdict:\n  within epoch: " + ("IDENTICAL" if h1 == h2 else "DIFFERENT"))

    if args.save_wavs:
        out = Path(args.save_wavs)
        _save_triad(out, "epoch1_batch1", _bt(clean_bt), _bt(noisy_bt))
        _save_triad(out, "epoch1_batch2", _bt(batch["clean"]), _bt(batch["noisy"]))
        print(f"[save] wrote WAV triads to {out.resolve()}")


def run_loader(args):
    print("\n" + "=" * 80 + "\n2) LOADER MODE\n" + "=" * 80)

    cfg = _load_yaml(Path(args.config))
    tr, va, _ = build_loaders(cfg)
    loader = tr
    ds = loader.dataset

    # attach callback with forced deterministic args
    cb = ProcNoiseAugmentCallback(
        sr=int(cfg["data"]["sr"]),
        prob=args.prob, snr_min=args.snr_min, snr_max=args.snr_max,
        fixed_seed=args.fixed_seed,
        fixed_per_epoch=args.fixed_per_epoch,
        train_only=True, track_stats=True,
        out_peak=(1.0 if args.no_peak else 0.98),
    )
    print(f"[cb] effective args: prob={cb.prob} snr_min={cb.snr_min} snr_max={cb.snr_max} "
          f"fixed_seed={cb.fixed_seed} fixed_per_epoch={cb.fixed_per_epoch} train_only={cb.train_only}")

    # pin-clean support: capture first clean and reuse on all batches
    pin_clean = bool(args.pin_clean)
    pinned_clean_bt = None
    pinned_noisy_bt = None

    # set fixed index if requested
    fixed_idx = args.fixed_index
    if fixed_idx is not None:
        # Quick way: bypass sampler and query the same item each time
        def fixed_iter():
            while True:
                yield fixed_idx
        index_iter = fixed_iter()
    else:
        index_iter = None

    hashes_by_epoch = {}
    snrs_by_epoch = {}

    def _next_item():
        if index_iter is None:
            return next(iter(loader))
        # manually fetch one index
        i = next(index_iter)
        item = ds[i]
        # collate: expecting (noisy, clean, ...)
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            noisy, clean = item[0], item[1]
        elif isinstance(item, dict):
            noisy, clean = item.get("noisy"), item.get("clean")
        else:
            raise RuntimeError("Unexpected dataset item structure")
        # add batch dim
        if noisy.dim() == 1: noisy = noisy.unsqueeze(0)
        if clean.dim() == 1: clean = clean.unsqueeze(0)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        return [noisy, clean]

    for ep in range(1, args.epochs + 1):
        hashes = []
        snrs = []
        cb.on_epoch_start(state=TrainState(epoch=ep))

        for b in range(args.batches):
            batch = _next_item()

            # convert to dict form the callback handles
            pack = {"noisy": batch[0], "clean": batch[1]}

            # pin the clean segment (and base noisy=clean) to remove dataset randomness
            if pin_clean:
                if pinned_clean_bt is None:
                    pinned_clean_bt = _bt(pack["clean"]).clone()
                    pinned_noisy_bt = pinned_clean_bt.clone()
                pack["clean"] = pinned_clean_bt.clone().unsqueeze(1)  # (B,1,T)
                pack["noisy"] = pinned_noisy_bt.clone().unsqueeze(1)

            # SNR before and after
            pre = _snr_db(_bt(pack["clean"]), _bt(pack["clean"]))
            print(f"[proc-noise] pre-mix SNR ~ +{pre:.2f} dB")

            cb.on_batch_start(trainer=None, state=TrainState(epoch=ep), batch=pack)

            post = _snr_db(_bt(pack["clean"]), _bt(pack["noisy"]))
            print(f"[proc-noise] post-mix SNR ~ +{post:.2f} dB")

            resid_ls = _ls_residual_unit_rms(_bt(pack["noisy"]), _bt(pack["clean"]))
            h = _hash_resid(resid_ls)
            hashes.append(h)
            snrs.append(post)

            if args.save_wavs:
                outdir = Path(args.save_wavs) / f"ep{ep:02d}"
                _save_triad(outdir, f"b{b:02d}", _bt(pack["clean"]), _bt(pack["noisy"]))

        hashes_by_epoch[ep] = hashes
        snrs_by_epoch[ep] = snrs

    print("\nResidual hashes per epoch:")
    for ep, H in hashes_by_epoch.items():
        print(f"  epoch {ep:02d}:")
        for i, h in enumerate(H):
            print(f"    b{i:02d}: {h}")

    print("\nSNR dB per epoch (mean over B):")
    for ep, S in snrs_by_epoch.items():
        print(f"  epoch {ep:02d}: " + ", ".join(f"{s:+.2f}" for s in S))

    def verdict(H):
        return "IDENTICAL" if len(set(H)) == 1 else "DIFFERENT"

    print("\nDeterminism verdict (loader):")
    for ep, H in hashes_by_epoch.items():
        print(f"  within epoch {ep:02d}: {verdict(H)}")
    if args.epochs >= 2:
        print(f"  across epochs (first batch): " +
              ("IDENTICAL" if hashes_by_epoch[1][0] == hashes_by_epoch[2][0] else "DIFFERENT"))

    if args.save_wavs:
        print(f"[save] wrote WAV triads to {Path(args.save_wavs).resolve()}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "loader"], default="synthetic")
    ap.add_argument("--config", type=str, default="", help="YAML, for loader mode")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--fixed-index", type=int, default=None, help="Use a constant dataset index")
    ap.add_argument("--pin-clean", action="store_true", help="Reuse the first clean for all batches (loader)")
    ap.add_argument("--no-peak", action="store_true", help="Force out_peak=1.0 in callback")

    ap.add_argument("--prob", type=float, default=1.0)
    ap.add_argument("--snr_min", type=float, default=8.0)
    ap.add_argument("--snr_max", type=float, default=8.0)
    ap.add_argument("--fixed_seed", type=int, default=1234)
    ap.add_argument("--fixed_per_epoch", action="store_true")

    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--save-wavs", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_loader(args)


if __name__ == "__main__":
    main()
