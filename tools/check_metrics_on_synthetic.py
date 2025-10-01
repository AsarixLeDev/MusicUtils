import torch, math
from soundrestorer.metrics.common import si_sdr_db

def snr_db(x, y):
    # SNR(noisy=y, clean=x) as used in audit: 10log10(||x||^2 / ||y-x||^2)
    eps = 1e-12
    num = (x**2).sum().clamp_min(eps)
    den = ((y - x)**2).sum().clamp_min(eps)
    return float(10.0 * torch.log10(num/den))

torch.manual_seed(0)
sr = 48000
T  = 3 * sr
clean = torch.randn(T)
for target in [3, 6, 10, 20]:
    # mix noise to the desired SNR
    n = torch.randn_like(clean)
    Pc = clean.pow(2).mean()
    Pn = n.pow(2).mean()
    alpha = torch.sqrt(Pc / (Pn * (10**(target/10))))
    noisy = clean + alpha * n

    # report
    s1 = snr_db(clean, noisy)
    s2 = float(si_sdr_db(noisy.unsqueeze(0), clean.unsqueeze(0), match_length=True).mean())
    print(f"target SNR={target:>2} dB | measured SNR={s1:6.2f} dB | SI-SDR(noisy,clean)={s2:6.2f} dB")
