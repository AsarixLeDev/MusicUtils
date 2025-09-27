import torch, contextlib
from ..core.registry import MODELS
from .complex_unet_lstm import ComplexUNetLSTM
from .complex_unet import ComplexUNetWrapper as ComplexUNet

@MODELS.register("complex_unet_auto")
class AutoComplexUNet(torch.nn.Module):
    def __init__(self, base=48, prefer_temporal=True, min_free_mem_mb=3000,
                 lstm_hidden=128, lstm_layers=2, bidirectional=True):
        super().__init__()
        want_lstm = bool(prefer_temporal)
        if want_lstm and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_mb = free // (1024*1024)
            if free_mb >= int(min_free_mem_mb):
                try:
                    self.inner = ComplexUNetLSTM(base=base, lstm_hidden=lstm_hidden,
                                                 lstm_layers=lstm_layers, bidirectional=bidirectional)
                    return
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower(): raise
        # fallback
        self.inner = ComplexUNet(base=base)

    def forward(self, x):  # (B,2,F,T)
        return self.inner(x)
