# -*- coding: utf-8 -*-
from .l1_wave import L1WaveLoss
from .mrstft import MRSTFTLoss
from .sisdr_pos import SISDRPositiveLoss

LOSS_REGISTRY = {
    "l1_wave": L1WaveLoss,
    "mrstft": MRSTFTLoss,
    "sisdr_pos": SISDRPositiveLoss,
    # convenient aliases
    "si_sdr_pos": SISDRPositiveLoss,
    "sisdr_ratio": SISDRPositiveLoss,  # same objective (error ratio)
}


def build_loss(name: str, args=None):
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss name: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    cls = LOSS_REGISTRY[name]
    return cls(**(args or {}))
