from __future__ import annotations
import torch
from typing import Dict, Any, Tuple
from ..utils.audio import hann_window, stft_pair, istft_from
from ..core.registry import TASKS

@TASKS.register("denoise_stft")
class DenoiseSTFTTask:
    """
    Turns (noisy, clean) -> model input (RI), applies mask mapping,
    reconstructs time-domain yhat. Returns outputs dict + per-sample proxy (L1).
    """
    def __init__(self, n_fft=1024, hop=256, mask_variant="plain", mask_limit=0.0,
                 clamp_mask_tanh: float = 0.0,  # NEW: 0 disables; else tanh clamp +/-clamp
                 safe_unity_fallback: bool = True,  # NEW
                 device="cuda"):
        self.n_fft=int(n_fft); self.hop=int(hop)
        self.mask_variant=str(mask_variant).lower()
        self.mask_limit=float(mask_limit)
        self.clamp_mask = float(clamp_mask_tanh)
        self.safe_unity = bool(safe_unity_fallback)
        self.device=device
        self._win = None

    def _win_on(self, device):
        if (self._win is None) or (self._win.device != device):
            from soundrestorer.utils.audio import hann_window
            self._win = hann_window(self.n_fft, device)
        # ensure normal (non-inference) tensor
        return self._win.clone().detach()

    def _map_mask(self, Mr: torch.Tensor, Mi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mask_variant == "mag_sigm1":
            Mag = 1.0 + (torch.sigmoid(Mr) - 0.5)
            R, I = Mag, torch.zeros_like(Mag)
        elif self.mask_variant == "mag":
            Mag = torch.sqrt(Mr**2 + Mi**2 + 1e-8)
            R, I = Mag, torch.zeros_like(Mag)
        elif self.mask_variant == "delta1":
            R, I = 1.0 + Mr, Mi
        elif self.mask_variant == "mag_delta1":
            Mag = 1.0 + torch.sqrt(Mr**2 + Mi**2 + 1e-8)
            R, I = Mag, torch.zeros_like(Mag)
        else:  # "plain"
            R, I = Mr, Mi

        if self.mask_limit > 0:
            mag_eff = torch.sqrt(R**2 + I**2 + 1e-8)
            scale = torch.clamp(self.mask_limit / mag_eff, max=1.0)
            R, I = R*scale, I*scale
        return R, I

    def step(self, model, batch):
        # unpack
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]
            meta = batch[3] if (len(batch) >= 4) else None
        else:
            raise RuntimeError("Batch must be (noisy, clean, extra?, meta?)")

        # use the model’s param device/dtype as the source of truth
        p = next(model.parameters())
        p_device, p_dtype = p.device, p.dtype
        noisy = noisy.to(p_device, non_blocking=True)
        clean = clean.to(p_device, non_blocking=True)

        win = self._win_on(p_device)
        Xn, Xn_ri = stft_pair(noisy, win, self.n_fft, self.hop)  # float32
        if Xn_ri.dtype != p_dtype:
            Xn_ri = Xn_ri.to(p_dtype)
        M = model(Xn_ri)  # (B,2,F,T)
        Mr, Mi = M[:, 0].float(), M[:, 1].float()
        if self.clamp_mask > 0.0:
            Mr = torch.tanh(Mr) * self.clamp_mask
            Mi = torch.tanh(Mi) * self.clamp_mask
        Xr, Xi = Xn.real.float(), Xn.imag.float()
        R, I = self._map_mask(Mr, Mi)
        Xhat = torch.complex(R * Xr - I * Xi, R * Xi + I * Xr)
        yhat = istft_from(Xhat, win, length=noisy.shape[-1], n_fft=self.n_fft, hop=self.hop)  # (B,T)

        if self.safe_unity and not torch.isfinite(yhat).all():
            with torch.no_grad():
                bad = ~torch.isfinite(yhat).all(dim=-1)  # (B,)
                if bad.any():
                    # exact-unity reconstruction for the bad samples only
                    y_unity = istft_from(Xn, win, length=noisy.shape[-1], n_fft=self.n_fft, hop=self.hop)
                    yhat[bad] = y_unity[bad]

        # per-sample proxy
        y_m = yhat if yhat.dim() == 2 else yhat.mean(dim=1)
        c_m = clean.float() if clean.dim() == 2 else clean.float().mean(dim=1)
        per_sample = torch.mean(torch.abs(y_m - c_m), dim=-1)

        outputs = {
            "yhat": yhat, "noisy": noisy, "clean": clean,
            "R": R, "I": I, "Mr": Mr, "Mi": Mi,
            "Xn": Xn, "Xhat": Xhat,
            "ids": (meta.get("id") if isinstance(meta, dict) else None)
        }
        return outputs, per_sample

