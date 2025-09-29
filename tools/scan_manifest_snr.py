#!/usr/bin/env python3
import argparse, random
import torch
from pathlib import Path
from soundrestorer.core.config import load_yaml
from soundrestorer.data.builder import build_denoise_loader

def mono_1d(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() == 1:
        return t
    if t.dim() == 2:  # (C,T)
        return t.mean(dim=0)
    raise RuntimeError(f"bad shape {tuple(t.shape)}")

def snr_db(clean, noisy, eps=1e-12):
    n = min(clean.numel(), noisy.numel())
    c = clean[:n]; y = noisy[:n]
    num = torch.sum(c**2).clamp_min(eps)
    den = torch.sum((y - c)**2).clamp_min(eps)
    return float(10.0 * torch.log10(num/den))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train","val"], default="val")
    ap.add_argument("--num", type=int, default=200, help="number of items to sample")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = cfg["data"]
    manifest = data["val_manifest"] if args.split=="val" else data["train_manifest"]

    ds, ld, _ = build_denoise_loader(manifest, data, train=(args.split=="train"))
    N = len(ds)
    print(f"{args.split} dataset size: {N}")

    rng = random.Random(args.seed)
    idxs = [rng.randrange(0, N) for _ in range(min(args.num, N))]

    snrs = []
    sil_n = sil_c = 0

    def silence_frac(w, thr_dbfs=-60.0):
        thr = 10.0**(thr_dbfs/20.0)
        w = torch.as_tensor(w, dtype=torch.float32)
        if w.dim()==2: w=w.mean(dim=0)
        return float((torch.abs(w) < thr).float().mean())

    for i in idxs:
        item = ds[i]
        if not (isinstance(item, (list,tuple)) and len(item)>=2):
            continue
        noisy = mono_1d(item[0]); clean = mono_1d(item[1])
        snrs.append(snr_db(clean, noisy))
        sil_n += float(silence_frac(noisy) > 0.95)
        sil_c += float(silence_frac(clean) > 0.95)

    if not snrs:
        print("No pairs collected.")
        return

    import statistics as st
    print(f"SNR(noisy,clean) {args.split}: "
          f"mean={st.mean(snrs):+.2f} dB | median={st.median(snrs):+.2f} dB | "
          f"min={min(snrs):+.2f} | max={max(snrs):+.2f}")
    print(f" >25 dB (too clean): {sum(1 for v in snrs if v>25.0)}/{len(snrs)}")
    print(f" silence>95%: noisy={int(sil_n)}/{len(snrs)} | clean={int(sil_c)}/{len(snrs)}")

if __name__ == "__main__":
    main()
