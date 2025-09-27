"""
AudioRestorer: training & inference utilities for spectral denoising.
"""
__version__ = "0.1.0"

from .losses.mrstft import MultiResSTFTLoss  # noqa: F401
# Lightweight re-exports (okay to import at top-level)
from .models.denoiser_net import ComplexUNet  # noqa: F401

__all__ = [
    "ComplexUNet",
    "MultiResSTFTLoss",
    "__version__",
]
