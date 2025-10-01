from __future__ import annotations
import torch
from typing import Tuple
from ..utils.audio import stft_pair, istft_from
from ..core.registry import TASKS

@TASKS.register("denoise_stft")
class DenoiseSTFTTask:
    def __init__(self, n_fft=1024, hop=256, mask_variant="plain", mask_limit=0.0,
                 clamp_mask_tanh: float = 0.0,
                 safe_unity_fallback: bool = True,
                 device="cuda",
                 mask_floor: float = 0.50):                 # <-- NEW
        self.n_fft=int(n_fft); self.hop=int(hop)
        self.mask_variant=str(mask_variant).lower()
        self.mask_limit=float(mask_limit)
        self.clamp_mask = float(clamp_mask_tanh)
        self.safe_unity = bool(safe_unity_fallback)
        self.device=device
        self.mask_floor = float(mask_floor)               # <-- NEW
        self._win = None

    def _win_on(self, device, dtype=torch.float32):
        """
        Cache a Hann window on the right device/dtype.
        Trainer clears self._win after warmup to rebuild under normal autograd.
        """
        if (self._win is None) or (self._win.device != device) or (self._win.dtype != dtype):
            self._win = torch.hann_window(self.n_fft, device=device, dtype=dtype)
        return self._win  # no clone: reuse the buffer

    def _map_mask(self, Mr: torch.Tensor, Mi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.mask_variant
        if v == "mag_sigm1":
            Mag = 1.0 + (torch.sigmoid(Mr) - 0.5)  # 0.5 residual range
            R, I = Mag, torch.zeros_like(Mag)
        elif v == "mag":
            # base magnitude-only (gradient ok everywhere)
            Mag = torch.relu(Mr) + 1.0  # softplus-like but cheaper
            R, I = Mag, torch.zeros_like(Mag)
        elif v == "delta1":
            R, I = 1.0 + Mr, Mi  # complex residual (may rotate phase)
        elif v == "mag_delta1":
            # >>> FIX: residual magnitude with non-zero grad at 0
            # ignore Mi for magnitude; let phase come only from Xn
            Mag = 1.0 + Mr  # gradient @ 0 is 1.0
            R, I = Mag, torch.zeros_like(Mag)
        elif v == "mag_sigmoid":
            # smooth, bounded magnitude mask in [mask_floor, mask_limit]
            # assumes mask_floor < mask_limit and both > 0
            floor = torch.tensor(getattr(self, "mask_floor", 0.5), device=Mr.device, dtype=Mr.dtype)
            limit = torch.tensor(getattr(self, "mask_limit", 1.6), device=Mr.device, dtype=Mr.dtype)
            width = torch.clamp(limit - floor, min=1e-3)
            M = floor + width * torch.sigmoid(Mr)  # Mr is raw head output (tanh-clamped if you keep it)
            R, I = M, torch.zeros_like(M)
        else:  # "plain"
            R, I = Mr, Mi

        # two-sided clamp on applied magnitude: M in [mask_floor, mask_limit]
        mag_eff = torch.sqrt(R ** 2 + I ** 2 + 1e-8)
        scale = torch.ones_like(mag_eff)
        if getattr(self, "mask_floor", 0.0) > 0.0:
            scale = torch.where(mag_eff < self.mask_floor,
                                self.mask_floor / (mag_eff + 1e-8),
                                scale)
        if getattr(self, "mask_limit", 0.0) > 0.0:
            scale = torch.where(mag_eff > self.mask_limit,
                                self.mask_limit / (mag_eff + 1e-8),
                                scale)
        R, I = R * scale, I * scale
        return R, I

    def step(self, model, batch):
        # unpack
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise RuntimeError("Batch must be (noisy, clean, [extra], [meta])")
        noisy, clean = batch[0], batch[1]
        meta = batch[3] if (len(batch) >= 4) else None

        # use the model’s param device/dtype as the source of truth
        p = next(model.parameters())
        p_device, p_dtype = p.device, p.dtype
        noisy = noisy.to(p_device, non_blocking=True)
        clean = clean.to(p_device, non_blocking=True)

        # STFT in a numerically-stable dtype; most torch.stft paths are float32 anyway.
        # Keep the window in float32; Xn will come back as complex64.
        win = self._win_on(p_device, dtype=torch.float32)
        Xn, Xn_ri = stft_pair(noisy, win, self.n_fft, self.hop)  # Xn: complex64, Xn_ri: float32
        # Feed model in its compute dtype (AMP-friendly)
        if Xn_ri.dtype != p_dtype:
            Xn_ri = Xn_ri.to(p_dtype)

        M = model(Xn_ri)  # (B,2,F,T) in p_dtype under autocast
        Mr, Mi = M[:, 0], M[:, 1]

        # Optional clamp for stability; stay in model dtype
        if self.clamp_mask > 0.0:
            Mr = torch.tanh(Mr) * self.clamp_mask
            Mi = torch.tanh(Mi) * self.clamp_mask

        # Mix in float32 with STFT (complex64) for numeric stability
        Mr32 = Mr.to(torch.float32)
        Mi32 = Mi.to(torch.float32)
        Xr = Xn.real.to(torch.float32)
        Xi = Xn.imag.to(torch.float32)
        R, I = self._map_mask(Mr32, Mi32)
        Xhat = torch.complex(R * Xr - I * Xi, R * Xi + I * Xr)

        yhat = istft_from(Xhat, win, length=noisy.shape[-1], n_fft=self.n_fft, hop=self.hop)  # (B,T), float32

        if self.safe_unity and not torch.isfinite(yhat).all():
            with torch.no_grad():
                bad = ~torch.isfinite(yhat).all(dim=-1)  # (B,)
                if bad.any():
                    # exact-unity reconstruction for the bad samples only
                    y_unity = istft_from(Xn, win, length=noisy.shape[-1], n_fft=self.n_fft, hop=self.hop)
                    yhat[bad] = y_unity[bad]

        # per-sample proxy (mean L1 over time)
        y_m = yhat if yhat.dim() == 2 else yhat.mean(dim=1)
        c_m = clean.to(torch.float32) if clean.dim() == 2 else clean.to(torch.float32).mean(dim=1)
        per_sample = torch.mean(torch.abs(y_m - c_m), dim=-1)

        outputs = {
            "yhat": yhat, "noisy": noisy, "clean": clean,
            "R": R, "I": I, "Mr": Mr32, "Mi": Mi32,
            "Xn": Xn, "Xhat": Xhat,
            "ids": (meta.get("id") if isinstance(meta, dict) else None)
        }
        return outputs, per_sample
