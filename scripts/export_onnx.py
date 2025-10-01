# scripts/export_onnx.py
import argparse, yaml, torch
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

class TaskWrapper(torch.nn.Module):
    def __init__(self, model, task, sr=48000):
        super().__init__()
        self.model = model
        self.task  = task
        self.sr    = sr
    def forward(self, x):                       # x: [B,1,T] waveform
        batch = {"noisy": x, "sr": self.sr}
        out = self.task.forward(self.model, batch)
        return out["yhat"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--length", type=int, default=48000)  # 1s
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    fac = _try_factories()
    assert "build_model" in fac and "build_task" in fac, "Factory not found"

    model = fac["build_model"](cfg["model"])
    task  = fac["build_task"](cfg["task"])
    sd = torch.load(args.checkpoint, map_location="cpu")
    sd_model = sd.get("model", sd)
    model.load_state_dict(sd_model, strict=False)
    model.eval()

    wrapper = TaskWrapper(model, task, sr=args.sr)
    x = torch.randn(1,1,args.length, dtype=torch.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, x, args.out,
        input_names=["noisy"], output_names=["yhat"],
        dynamic_axes={"noisy": {2: "time"}, "yhat": {2: "time"}},
        opset_version=17
    )
    print(f"[onnx] exported to {args.out}")

if __name__ == "__main__":
    main()
