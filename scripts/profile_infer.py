# scripts/profile_infer.py
import argparse, time, torch, yaml
from pathlib import Path

def _try_factories():
    out = {}
    for mod, fn in [("soundrestorer.core.factory", "build_model"),
                    ("soundrestorer.core.factory", "build_task")]:
        try:
            m = __import__(mod, fromlist=[fn])
            out[fn] = getattr(m, fn)
        except Exception:
            pass
    return out

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    fac = _try_factories()
    model = fac["build_model"](cfg["model"])
    task  = fac["build_task"](cfg["task"])
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.to(args.device).eval()

    T = int(args.secs * args.sr)
    x = torch.randn(1,1,T, device=args.device)

    # warmup
    for _ in range(args.warmup):
        batch = {"noisy": x, "sr": args.sr}
        _ = task.forward(model, batch)["yhat"]
        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.synchronize()

    # measure
    times = []
    max_mem = 0
    for _ in range(args.runs):
        t0 = time.perf_counter()
        batch = {"noisy": x, "sr": args.sr}
        y = task.forward(model, batch)["yhat"]
        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        if torch.cuda.is_available() and "cuda" in args.device:
            mm = torch.cuda.max_memory_allocated(device=args.device)
            max_mem = max(max_mem, mm)

    import statistics as stat
    print(f"[infer] median {stat.median(times)*1000:.1f} ms | mean {stat.mean(times)*1000:.1f} ms | p90 {stat.quantiles(times, n=10)[8]*1000:.1f} ms")
    if max_mem:
        print(f"[infer] peak mem {max_mem/1e6:.1f} MB")

if __name__ == "__main__":
    main()
