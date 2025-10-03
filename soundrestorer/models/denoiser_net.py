from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ComplexUNet"]


# ---------- building blocks ----------

class ConvBNAct(nn.Module):
    """
    Named submodules (.conv, .bn) so state_dict keys look like:
      enc1.net.0.conv.weight, enc1.net.0.bn.weight, ...
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class EncBlock(nn.Module):
    """Two convs; downsampling is done outside (MaxPool2d or stride-2)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """
    Upsample -> concat skip -> two convs
    Submodule named 'conv' for compatibility with keys like up4.conv.0.conv.weight
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # resize x to skip's spatial size
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------- Complex UNet ----------

class ComplexUNet(nn.Module):
    """
    U-Net operating on real/imag channels of an STFT: input [B, 2, F, T] -> output [B, 2, F, T] (complex mask).
    Channel plan (for base=48) matches your historical checkpoints:
      enc1: 48, enc2: 96, enc3: 192, enc4: 384, bottleneck: 384
      up4 -> 192, up3 -> 96, up2 -> 48, up1 -> 24, out 1x1 -> 2
    This reproduces state_dict keys like:
      enc1.net.0.conv.weight, up4.conv.0.bn.weight, out.weight, ...
    """

    def __init__(self, base: int = 48, **_):
        super().__init__()
        b1 = base
        b2 = base * 2
        b3 = base * 4
        b4 = base * 8
        b5 = b4  # bottleneck width
        b0 = max(2, base // 2)  # last decoder stage (e.g., 24 for base=48)

        # encoder
        self.enc1 = EncBlock(2, b1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = EncBlock(b1, b2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = EncBlock(b2, b3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = EncBlock(b3, b4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck (two convs, keep same width — matches your checkpoint shapes)
        self.bottleneck = nn.Sequential(
            ConvBNAct(b4, b5, 3, 1, 1),
            ConvBNAct(b5, b5, 3, 1, 1),
        )

        # decoder
        self.up4 = UpBlock(in_ch=b5 + b4, out_ch=b3)  # (384+384)->192 for base=48
        self.up3 = UpBlock(in_ch=b3 + b3, out_ch=b2)  # (192+192)->96
        self.up2 = UpBlock(in_ch=b2 + b2, out_ch=b1)  # (96+96)->48
        self.up1 = UpBlock(in_ch=b1 + b1, out_ch=b0)  # (48+48)->24

        # output 1x1 conv to 2 channels (real & imag mask)
        self.out = nn.Conv2d(b0, 2, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # SiLU ≈ ReLU for init purposes → use He/Kaiming (fan_in, ReLU gain)
                try:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                except Exception:
                    # very old torch fallback
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, F, T]
        returns: [B, 2, F, T] complex mask (real, imag)
        """
        # encoder
        e1 = self.enc1(x)  # B,b1,F,T
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)  # B,b2,..
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)  # B,b3,..
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)  # B,b4,..
        p4 = self.pool4(e4)

        # bottleneck
        bn = self.bottleneck(p4)  # B,b5,..

        # decoder with skip connections
        d4 = self.up4(bn, e4)  # -> b3
        d3 = self.up3(d4, e3)  # -> b2
        d2 = self.up2(d3, e2)  # -> b1
        d1 = self.up1(d2, e1)  # -> b0

        out = self.out(d1)  # -> 2 channels
        return out
