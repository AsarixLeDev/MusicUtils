import torch
from soundrestorer.utils import ensure_3d, to_mono, si_sdr_db, snr_db
x = torch.randn(48000)
y = torch.randn(1,48000)
z = torch.randn(2,48000)
assert ensure_3d(x).shape == (1,1,48000)
assert to_mono(z).shape == (1,1,48000)
clean = torch.randn(1,1,48000)
noisy = clean + 0.1*torch.randn_like(clean)
yhat = clean + 0.05*torch.randn_like(clean)
snr1 = snr_db(noisy, clean)[0]
snr2 = snr_db(yhat, clean)[0]
si1 = si_sdr_db(noisy, clean)[0]
si2 = si_sdr_db(yhat, clean)[0]
assert snr2 >= snr1 and si2 >= si1
from soundrestorer.utils import add_noise_at_snr, snr_db
c = torch.randn(1,1,48000)
n = torch.randn(1,1,48000)
mix = add_noise_at_snr(c, n, snr_db=6.0)
measured = snr_db(mix, c)[0].item()
# should be ~6 dB within small tolerance
