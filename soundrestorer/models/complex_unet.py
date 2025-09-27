from ..core.registry import MODELS
# Reuse your existing ComplexUNet class
from soundrestorer.models.denoiser_net import ComplexUNet as _ComplexUNet
import torch.nn as nn

@MODELS.register("complex_unet")
class ComplexUNetWrapper(_ComplexUNet):
    # you can override __init__ to adapt args if needed
    def __init__(self, base=48, **kwargs):
        super().__init__(base=int(base), **kwargs)
