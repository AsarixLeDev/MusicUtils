import torch
import torch.nn as nn

from soundrestorer.models.denoiser_net import ComplexUNet as _ComplexUNet
from ..core.registry import MODELS


@MODELS.register("complex_unet_lstm")
class ComplexUNetLSTM(nn.Module):
    def __init__(self, base=48, lstm_hidden=128, lstm_layers=2, bidirectional=True, dropout=0.0):
        super().__init__()
        self.net = _ComplexUNet(base=int(base))  # stays fp32 by default
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)
        self._built = False  # build when we know F
        # placeholders for torchscript friendliness
        self.rnn = None
        self.proj = None

    def _build_once(self, F, ref_device):
        D = 2 * F  # input features = (Re,Im) per freq bin
        H = self.lstm_hidden
        rnn = nn.LSTM(
            input_size=D, hidden_size=H, num_layers=self.lstm_layers,
            batch_first=True, bidirectional=self.bidirectional,
            dropout=(self.dropout if self.lstm_layers > 1 else 0.0)
        )
        out_dim = H * (2 if self.bidirectional else 1)
        proj = nn.Linear(out_dim, D)
        # Important: keep LSTM weights float32 (more stable), but move to the same device
        self.rnn = rnn.to(device=ref_device, dtype=torch.float32)
        self.proj = proj.to(device=ref_device, dtype=torch.float32)
        self._built = True

    def forward(self, Xri):  # (B,2,F,T)
        M = self.net(Xri)  # (B,2,F,T), fp32
        B, C, F, T = M.shape
        if not self._built:
            self._build_once(F, ref_device=M.device)

        # (B,2,F,T) -> (B,T,2F)
        z = M.permute(0, 3, 1, 2).contiguous().view(B, T, 2 * F)
        z, _ = self.rnn(z)  # (B,T,H*dir)
        z = self.proj(z)  # (B,T,2F)
        z = z.view(B, T, 2, F).permute(0, 2, 3, 1).contiguous()  # (B,2,F,T)
        return z
