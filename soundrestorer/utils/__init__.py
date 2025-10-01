# -*- coding: utf-8 -*-
# soundrestorer/utils/__init__.py

from .audio import (
    ensure_3d, to_mono, normalize_peak, pad_or_trim, match_length,
    random_time_crop_pair, random_gain_db, is_silent, peak,
)
from .io import (
    read_wav, write_wav, read_jsonl, write_jsonl, write_csv,
)
from .metrics import (
    rms, rms_db, snr_db, si_sdr_db, mae, mse,
)
from .signal import (
    amp_to_db, db_to_amp, pow_to_db, add_noise_at_snr,
    stft_complex, istft_complex,
)
from .torch_utils import (
    move_to, autocast_from_amp, need_grad_scaler, set_channels_last,
    strip_state_dict_prefixes, load_state_dict_loose, latest_checkpoint,
    format_ema, set_seed, count_parameters,
)

__all__ = [
    # audio
    "ensure_3d", "to_mono", "normalize_peak", "pad_or_trim", "match_length",
    "random_time_crop_pair", "random_gain_db", "is_silent", "peak",
    # metrics
    "rms", "rms_db", "snr_db", "si_sdr_db", "mae", "mse",
    # signal
    "amp_to_db", "db_to_amp", "pow_to_db", "add_noise_at_snr",
    "stft_complex", "istft_complex",
    # io
    "read_wav", "write_wav", "read_jsonl", "write_jsonl", "write_csv",
    # torch utils
    "move_to", "autocast_from_amp", "need_grad_scaler", "set_channels_last",
    "strip_state_dict_prefixes", "load_state_dict_loose", "latest_checkpoint",
    "format_ema", "set_seed", "count_parameters",
]
