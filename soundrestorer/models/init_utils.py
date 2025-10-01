# soundrestorer/models/init_utils.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


def _find_mask_head_conv(model: nn.Module) -> nn.Conv2d | None:
    """
    Heuristic: return the *last* Conv2d layer with out_channels == 2.
    Your denoiser head usually outputs (Mr, Mi) or 2-channel magnitude params.
    """
    head = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and getattr(m, "out_channels", None) == 2:
            head = m
    return head


def init_head_for_mask(
    model: nn.Module,
    mask_variant: str = "plain",
    mask_floor: float | None = None,
    mask_limit: float | None = None,
) -> bool:
    """
    Initialize the denoiser head so the *applied* mask starts at UNITY.
    - plain       : complex mask directly (R=Mr, I=Mi) -> bias Re=+1, Im=0
    - delta1      : complex residual (R=1+Mr, I=Mi)    -> bias Re=0,  Im=0
    - mag         : magnitude-only mask M = |(Mr,Mi)|  -> set Mr bias=1, Mi=0 (M≈1)
    - mag_delta1  : M = 1 + |(Mr,Mi)|                  -> bias zeros (M≈1 after clamp)
    - mag_sigm1   : M = 1 + (sigmoid(Mr)-0.5)          -> bias Mr=0 (M≈1)
    - mag_sigmoid : M = floor + (limit-floor)*sigmoid(Mr)
                    -> choose Mr bias so M==1 at init

    Returns True if initialization was applied, False if no suitable head was found.
    """
    mv = str(mask_variant).lower()
    head = _find_mask_head_conv(model)
    if head is None:
        print("[init] no Conv2d(out_channels=2) head found; skipping.")
        return False

    # Ensure bias exists
    if head.bias is None:
        head.bias = nn.Parameter(torch.zeros(2, device=head.weight.device, dtype=head.weight.dtype))

    with torch.no_grad():
        # Start from pure bias (no linear mixing initially)
        head.weight.zero_()
        # Two-element bias: channel 0 = Mr-like, channel 1 = Mi-like
        if mv == "plain":
            # Unity complex mask: R=1, I=0
            head.bias.zero_()
            head.bias[0] = 1.0
        elif mv == "delta1":
            # R = 1 + Mr, I = Mi -> Mr,Mi bias at 0 yields unity
            head.bias.zero_()
        elif mv == "mag":
            # M = sqrt(Mr^2 + Mi^2); pick Mr bias=1, Mi=0 so M≈1
            head.bias.zero_()
            head.bias[0] = 1.0
        elif mv == "mag_delta1":
            # M = 1 + sqrt(Mr^2 + Mi^2); bias zeros -> M≈1
            head.bias.zero_()
        elif mv == "mag_sigm1":
            # M = 1 + (sigmoid(Mr) - 0.5); Mr=0 -> M≈1
            head.bias.zero_()
        elif mv == "mag_sigmoid":
            # M = floor + (limit - floor) * sigmoid(Mr)
            # Choose bias so M == 1 at init:
            #   sigmoid(b) = (1 - floor) / (limit - floor)
            floor = 0.5 if mask_floor is None else float(mask_floor)
            limit = 1.6 if mask_limit is None else float(mask_limit)
            width = max(1e-6, limit - floor)
            p = (1.0 - floor) / width
            # Clamp p into (0,1) for numerical safety
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            bias_mr = math.log(p / (1.0 - p))  # logit
            head.bias.zero_()
            head.bias[0] = bias_mr
        else:
            # Default: safe unity for unknown variants
            head.bias.zero_()
            head.bias[0] = 1.0

    # Log final bias (cpu list for readability)
    b = head.bias.detach().cpu().tolist()
    print(f"[init] head bias set for mask_variant={mv}: bias=[{b[0]:.4f}, {b[1]:.4f}]")
    return True
