# tests/test_mixer.py
import torch
from soundrestorer.utils.signal import add_noise_at_snr
from soundrestorer.utils.metrics import snr_db

def test_snr_targets_and_silence():
    torch.manual_seed(0)
    sr = 8000
    T = 2 * sr
    clean = (torch.randn(1,1,T) * 0.1).clamp(-1,1)
    noise = torch.randn_like(clean)

    for tgt in [0.0, 6.0, 12.0, 20.0]:
        mix = add_noise_at_snr(clean, noise, snr_db=tgt)
        meas = float(snr_db(mix, clean)[0].item())
        assert abs(meas - tgt) < 0.8  # within tolerance

    # Silent handling: if clean tiny, result ~noise power (no NaNs)
    clean_z = torch.zeros_like(clean)
    mix2 = add_noise_at_snr(clean_z, noise, snr_db=10.0)
    m2 = float((mix2**2).mean())
    n2 = float((noise**2).mean())
    assert abs(m2 - n2) / max(n2, 1e-9) < 0.2
