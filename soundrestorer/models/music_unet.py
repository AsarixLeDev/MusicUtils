# -*- coding: utf-8 -*-
"""
MusicUNet
- 2D UNet over (freq,time), downsampling only in freq (stride=(2,1))
- Residual conv blocks with GELU + GroupNorm
- Output channels = 1 (magnitude mask) or 2 (complex RI/delta)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch, torch.nn as nn
import torch.nn.functional as F

def same_pad_2d(k: int) -> int:
    return (k - 1) // 2

class ConvBlock(nn.Module):
    def __init__(self, c, ks=3, gn=8, p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, ks, padding=same_pad_2d(ks))
        self.gn1   = nn.GroupNorm(min(gn, c), c)
        self.conv2 = nn.Conv2d(c, c, ks, padding=same_pad_2d(ks))
        self.gn2   = nn.GroupNorm(min(gn, c), c)
        self.drop  = nn.Dropout2d(p) if p > 0 else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        return F.gelu(x + h)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.pool = nn.Conv2d(c_out, c_out, 3, stride=(2,1), padding=1)  # down F only

    def forward(self, x):
        x = F.gelu(self.conv(x))
        x = self.pool(x)
        return x

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2,1), mode="nearest")
        self.conv_in = nn.Conv2d(c_in + c_skip, c_out, 3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        # pad/crop if freq mismatch due to odd sizes
        if x.size(-2) != skip.size(-2):
            diff = skip.size(-2) - x.size(-2)
            x = F.pad(x, (0,0, 0, diff)) if diff > 0 else x[..., :skip.size(-2), :]
        x = torch.cat([x, skip], dim=1)
        return F.gelu(self.conv_in(x))

@dataclass
class MusicUNetCfg:
    in_ch: int = 1           # log|S|
    base: int = 48
    depth: int = 5
    gn: int = 8
    drop: float = 0.0
    out_ch: int = 1          # 1 (mag mask) or 2 (RI/delta)

class MusicUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = MusicUNetCfg(**kwargs)
        self.cfg = cfg

        cs: List[int] = [cfg.base * (2**i) for i in range(cfg.depth)]
        self.enc0 = nn.Conv2d(cfg.in_ch, cs[0], 3, padding=1)
        self.blocks_enc = nn.ModuleList([ConvBlock(c, ks=3, gn=cfg.gn, p=cfg.drop) for c in cs])
        self.downs = nn.ModuleList([Down(cs[i], cs[i+1]) for i in range(cfg.depth-1)])

        self.mid = ConvBlock(cs[-1], ks=3, gn=cfg.gn, p=cfg.drop)

        # decoder
        self.ups   = nn.ModuleList([Up(cs[i+1], cs[i], cs[i]) for i in reversed(range(cfg.depth-1))])
        self.blocks_dec = nn.ModuleList([ConvBlock(cs[i], ks=3, gn=cfg.gn, p=cfg.drop) for i in reversed(range(cfg.depth-1))])

        self.head = nn.Conv2d(cs[0], cfg.out_ch, 1)

    def forward(self, x):  # x: (B, 1, F, T)
        skips = []
        h = F.gelu(self.enc0(x))
        # enc
        for i in range(len(self.blocks_enc)):
            h = self.blocks_enc[i](h)
            if i < len(self.downs):
                skips.append(h)
                h = self.downs[i](h)
        # mid
        h = self.mid(h)
        # dec
        for up, blk, sk in zip(self.ups, self.blocks_dec, reversed(skips)):
            h = up(h, sk)
            h = blk(h)
        return self.head(h)
