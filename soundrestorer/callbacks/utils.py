# -*- coding: utf-8 -*-
# soundrestorer/callbacks/utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torchaudio

from soundrestorer.metrics.common import (
    match_len, resample_if_needed, si_sdr_db, snr_db
)


@dataclass
class Triad:
    clean: Optional[torch.Tensor]  # [C,T]
    noisy: Optional[torch.Tensor]  # [C,T]
    yhat: Optional[torch.Tensor]  # [C,T]
    sr: int


def _to_mono_ct(x: torch.Tensor) -> torch.Tensor:
    # x: (C,T) or (T,)
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2 and x.size(0) > 1:
        return x.mean(dim=0, keepdim=True)
    return x

def _resid_noise_ls(noisy_ct: torch.Tensor, clean_ct: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Least-squares residual of noisy vs clean.
    Returns mono (1,T) residual ~ noise with clean removed (scale-invariant).
    """
    n_ct, c_ct = match_len(_to_mono_ct(noisy_ct), _to_mono_ct(clean_ct))  # (1,T)
    # remove DC (helps perceived 'musiciness')
    n_ct = n_ct - n_ct.mean(dim=-1, keepdim=True)
    c_ct = c_ct - c_ct.mean(dim=-1, keepdim=True)
    # α = <n,c> / ||c||^2
    num = (n_ct * c_ct).sum(dim=-1, keepdim=True)
    den = (c_ct.pow(2).sum(dim=-1, keepdim=True))
    alpha = num / (den + eps)
    resid = n_ct - alpha * c_ct  # (1,T)
    return resid

def _prep_audio(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure float32 on CPU, clamp to [-1,1], channel-first [C,T].
    Accepts [T], [C,T], [B,C,T]; returns [C,T].
    """
    if x is None:
        return None
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 3:
        # choose the first item in batch
        x = x[0]
    x = x.detach()
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    elif x.dtype != torch.float32:
        x = x.to(torch.float32)
    if x.device.type != "cpu":
        x = x.cpu()
    x = torch.clamp(x, -1.0, 1.0)
    return x


def save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sr)



def _resid_error(clean_ct: torch.Tensor, yhat_ct: torch.Tensor) -> torch.Tensor:
    """
    Model error residual = clean - yhat, mono.
    """
    c_ct, y_ct = match_len(_to_mono_ct(clean_ct), _to_mono_ct(yhat_ct))
    return c_ct - y_ct

def save_wav_triads(
    out_dir: Path,
    base: str,
    triad: Triad,
    *,
    want_resid: bool = True,
    resid_mode: str = "noise",    # "noise" (noisy - alpha*clean) or "error" (clean - yhat)
) -> Dict[str, Path]:
    """
    Save _clean/_noisy/_yhat (+ _resid) as float32 WAVs.
    Residual:
      - "noise": least-squares projection residual (noisy − α·clean), mono -> true injected noise
      - "error": model error (clean − yhat), mono
    """
    paths: Dict[str, Path] = {}
    c = _prep_audio(triad.clean) if triad.clean is not None else None    # (C,T)
    n = _prep_audio(triad.noisy) if triad.noisy is not None else None    # (C,T)
    y = _prep_audio(triad.yhat)  if triad.yhat  is not None else None    # (C,T)
    sr = int(triad.sr)

    if c is not None:
        p = out_dir / f"{base}_clean.wav"
        save_wav(p, c, sr); paths["clean"] = p
    if n is not None:
        p = out_dir / f"{base}_noisy.wav"
        save_wav(p, n, sr); paths["noisy"] = p
    if y is not None:
        p = out_dir / f"{base}_yhat.wav"
        save_wav(p, y, sr); paths["yhat"] = p

    if want_resid:
        resid: Optional[torch.Tensor] = None
        try:
            if resid_mode == "noise" and (n is not None and c is not None):
                resid = _resid_noise_ls(n, c)          # (1,T)
            elif resid_mode == "error" and (c is not None and y is not None):
                resid = _resid_error(c, y)             # (1,T)
        except Exception:
            resid = None

        if resid is not None:
            resid = torch.clamp(resid, -1.0, 1.0)
            p = out_dir / f"{base}_resid.wav"
            save_wav(p, resid, sr); paths["resid"] = p

    return paths

# soundrestorer/callbacks/utils.py

def _as_float(x):
    import torch
    if isinstance(x, torch.Tensor):
        # mean() in case a batch sneaks through; detach+cpu for safety
        return float(x.detach().mean().cpu().item())
    return float(x)


def triad_metrics(
        yhat: torch.Tensor,
        clean: Optional[torch.Tensor],
        noisy: Optional[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute a minimal set of quick metrics from waveforms (channel-first).
    Always returns plain Python floats, never tensors.
    """
    out: Dict[str, float] = {}
    if clean is not None:
        out["si_yhat_clean_db"] = _as_float(si_sdr_db(yhat, clean))
    if noisy is not None and clean is not None:
        si_n = _as_float(si_sdr_db(noisy, clean))
        si_y = out.get("si_yhat_clean_db", _as_float(si_sdr_db(yhat, clean)))
        out["si_noisy_clean_db"] = si_n
        out["delta_si_db"] = float(si_y - si_n)
        out["snr_noisy_clean_db"] = _as_float(snr_db(noisy, clean))
    return out


def infer_sr(trainer) -> Optional[int]:
    # Best effort — your Trainer already knows data SR in several places
    # (data cfg, dataset, task), we try common fields.
    for k in ("sr", "sample_rate", "data_sr"):
        if hasattr(trainer, k):
            v = getattr(trainer, k)
            try:
                return int(v)
            except Exception:
                pass
    return None


def ensure_sr(x: torch.Tensor, sr_x: int, target_sr: Optional[int]) -> Tuple[torch.Tensor, int]:
    if target_sr is None or target_sr == sr_x:
        return x, sr_x
    return resample_if_needed(x, sr_x, target_sr), target_sr
