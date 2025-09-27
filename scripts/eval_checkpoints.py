#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from soundrestorer.data.dataset import DenoiseDataset, DenoiseConfig
# project imports
from soundrestorer.losses.mrstft import MultiResSTFTLoss
from soundrestorer.models.denoiser_net import ComplexUNet
from soundrestorer.train.config import load_yaml, deep_update, parse_dotlist
from soundrestorer.train.plotting import setup_matplotlib
from soundrestorer.train.utils_stft import stft_pair, istft_from, hann_window

torch.backends.cudnn.benchmark = True
USE_CHANNELS_LAST = True


def collate(batch):
    noisy = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))
    clean = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    noise = torch.from_numpy(np.stack([b[2] for b in batch], axis=0))
    return noisy, clean, noise


def build_eval_loader(val_manifest: str, sr: int, crop: float, batch: int, workers: int, prefetch: int, cache_gb: float,
                      no_cache: bool):
    cfg = DenoiseConfig(sample_rate=sr, crop_seconds=crop, seed=1, enable_cache=(not no_cache), cache_gb=cache_gb)
    ds = DenoiseDataset(val_manifest, cfg)
    ld = DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=workers,
        drop_last=False, pin_memory=True,
        prefetch_factor=(prefetch if workers > 0 else None),
        persistent_workers=(workers > 0),
        collate_fn=collate
    )
    return ds, ld


def _infer_base_from_state(sd: Dict[str, Any]) -> Optional[int]:
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and k.endswith("enc1.net.0.conv.weight") and v.ndim == 4:
            return int(v.shape[0])
    return None


def load_state_into(model: torch.nn.Module, ckpt_path: Path, strict=False) -> Dict[str, Any]:
    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = state.get("model", state) if isinstance(state, dict) else state
    # strip DP
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:   print(f"[warn] missing keys ({len(missing)}), showing first 6: {missing[:6]}")
    if unexpected: print(f"[warn] unexpected keys ({len(unexpected)}), first 6: {unexpected[:6]}")
    return state


def epoch_num(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def find_checkpoints(ckpt: Optional[str], ckpt_dir: Optional[str]) -> List[Path]:
    if ckpt:
        p = Path(ckpt);
        if not p.exists(): raise FileNotFoundError(p)
        return [p]
    base = Path(ckpt_dir) if ckpt_dir else Path("./runs")
    if base.is_file(): return [base]
    files = sorted(base.rglob("epoch_*.pt"), key=epoch_num)
    if not files: raise FileNotFoundError(f"No epoch_*.pt under {base}")
    return files


def evaluate_ckpt(ckpt: Path, va_ld: DataLoader, device: str, n_fft: int, hop: int, mask_limit: float,
                  model_base_cli: int, channels_last: bool, compile_flag: bool) -> Tuple[float, int]:
    # infer base
    raw = torch.load(str(ckpt), map_location="cpu")
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    inferred = _infer_base_from_state(sd)
    if inferred is None and isinstance(raw, dict):
        inferred = raw.get("args", {}).get("model", {}).get("base", None)
        if inferred is not None:
            inferred = int(inferred)
    base = model_base_cli if model_base_cli > 0 else (inferred or 48)

    net = ComplexUNet(base=base).to(device)
    if channels_last and device.startswith("cuda"):
        net = net.to(memory_format=torch.channels_last)
    _ = load_state_into(net, ckpt, strict=False)

    if compile_flag and hasattr(torch, "compile"):
        try:
            import triton  # noqa
            net = torch.compile(net, mode="reduce-overhead")
        except Exception:
            pass

    net.eval()
    mrstft = MultiResSTFTLoss().to(device)
    win = hann_window(n_fft, device=device, dtype=torch.float32)

    tot = 0.0
    with torch.no_grad():
        pbar = tqdm(va_ld, desc=f"valid {ckpt.stem}", leave=False, dynamic_ncols=True)
        for noisy, clean, _ in pbar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            Xn, Xn_ri = stft_pair(noisy, win, n_fft=n_fft, hop=hop)
            if channels_last and device.startswith("cuda"):
                Xn_ri = Xn_ri.contiguous(memory_format=torch.channels_last)

            M = net(Xn_ri)

            if mask_limit and mask_limit > 0:
                Mr_, Mi_ = M[:, 0], M[:, 1]
                mag_ = torch.sqrt(Mr_ ** 2 + Mi_ ** 2 + 1e-8)
                scale_ = torch.clamp(mask_limit / mag_, max=1.0)
                M = torch.stack([Mr_ * scale_, Mi_ * scale_], dim=1)

            Mr, Mi = M[:, 0].float(), M[:, 1].float()
            Xr, Xi = Xn.real.float(), Xn.imag.float()
            Xhat = torch.complex(Mr * Xr - Mi * Xi, Mr * Xi + Mi * Xr)
            yhat = istft_from(Xhat, win, length=noisy.shape[-1], n_fft=n_fft, hop=hop)

            mr_out = mrstft(yhat, clean.float())
            mr = mr_out[0] if isinstance(mr_out, tuple) else mr_out
            l1 = torch.mean(torch.abs(yhat - clean.float()))
            loss = mr + 0.5 * l1

            tot += float(loss.item())
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
    avg = tot / max(1, len(va_ld))
    return avg, base


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate denoiser checkpoints and plot loss vs epoch")
    ap.add_argument("--config", required=True, help="YAML config for data/runtime (sr, crop, n_fft/hop, etc.)")
    ap.add_argument("--set", nargs="*", default=[], help="Override config keys: key=value (dotlist)")
    ap.add_argument("--ckpt", default=None, help="Path to single epoch_XXX.pt")
    ap.add_argument("--ckpt-dir", default=None, help="Folder (or runs/*) to search epoch_*.pt recursively")
    ap.add_argument("--model-base", type=int, default=-1, help="UNet base channels; -1=auto from ckpt")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-file", type=str, default="")
    ap.add_argument("--mpl-backend", type=str, default="auto")
    return ap.parse_args()


def main():
    args = parse_args()

    # load config (no run dirs are created here)
    cfg = load_yaml(args.config)
    if args.set:
        cfg = deep_update(cfg, parse_dotlist(args.set))

    sr = int(cfg["data"]["sr"])
    crop = float(cfg["data"]["crop_seconds"])
    batch = int(args.batch or cfg["data"]["batch"])
    workers = int(args.workers or cfg["data"]["workers"])
    prefetch = int(cfg["data"]["prefetch"])
    cache_gb = float(cfg["data"]["cache_gb"])
    no_cache = bool(cfg["data"]["no_cache"])
    n_fft = int(cfg["inference_defaults"]["n_fft"])
    hop = int(cfg["inference_defaults"]["hop"])
    mask_limit = float(cfg["loss"]["mask_limit"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va_ld = build_eval_loader(cfg["data"]["val_manifest"], sr, crop, batch, workers, prefetch, cache_gb, no_cache)
    ckpts = find_checkpoints(args.ckpt, args.ckpt_dir or cfg["paths"].get("checkpoints", None))

    results: List[Tuple[int, str, float]] = []
    for p in ckpts:
        ep = epoch_num(p)
        loss, base = evaluate_ckpt(
            p, va_ld, device,
            n_fft=n_fft, hop=hop, mask_limit=mask_limit,
            model_base_cli=args.model_base, channels_last=args.channels_last,
            compile_flag=args.compile
        )
        print(f"{p.name}: avg_val_loss={loss:.6f}  (model_base={base})")
        results.append((ep, p.name, loss))

    if not results:
        print("No results.");
        return

    results.sort(key=lambda x: x[0])
    best = min(results, key=lambda x: x[2])
    print(f"\nBest: epoch {best[0]}  {best[1]}  loss {best[2]:.6f}")

    # plotting
    if args.plot or args.plot_file:
        plt, used_backend, ok = setup_matplotlib(args.mpl_backend)
        if plt is None:
            print("[plot] matplotlib not available; skipping.")
            return
        xs = [r[0] for r in results]
        ys = [r[2] for r in results]
        plt.figure(figsize=(8.5, 4.5))
        plt.plot(xs, ys, marker="o")
        plt.title("Validation loss vs epoch")
        plt.xlabel("epoch");
        plt.ylabel("MR-STFT + 0.5×L1")
        plt.grid(True, alpha=0.3)
        if args.plot_file:
            Path(args.plot_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot_file, dpi=120, bbox_inches="tight")
            print(f"[plot] saved {args.plot_file}")
        if args.plot:
            try:
                plt.show()
            except Exception:
                pass


if __name__ == "__main__":
    main()
