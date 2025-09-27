# soundrestorer

A modular, hardware-flexible audio restoration toolkit (denoise, declip, bandwidth extension, inpainting) with CPU,
CUDA, and Blackwell-ready paths.

> Project skeleton generated for initial development. See `README.md` sections below for environment setup on CPU/GPU (
> CUDA 12.8+ for Blackwell).

## Install (minimal)

We intentionally **do not pin PyTorch** in `pyproject.toml` to avoid wheel/cuda mismatches. Install the appropriate
build first, then install this package.

### CPU-only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### NVIDIA GPU (non-Blackwell, e.g., RTX 20/30/40 with CUDA 12.6)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

### NVIDIA Blackwell (RTX 50 series, CUDA 12.8+)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Optionally for ONNX Runtime (CPU/GPU) and TensorRT (for maximum inference speed):

```bash
pip install onnx onnxruntime onnxruntime-gpu tensorrt
```

## Quick check

```bash
python scripts/check_env.py
soundrestorer --help
```

## Why this structure?

- Modular models (`denoiser`, `declipper`, `bandwidth_extension`, `inpainting`)
- Hardware abstraction (`core/device.py`) to safely pick CPU/GPU and support sm_120 (Blackwell) without crashing.
- Export/optimize paths (`inference/export.py`, `inference/optimize_trt.py`) for ONNX/TensorRT.
- Robust I/O utilities and CLI that won't break if optional deps are missing.

## License

MIT
