# tests/test_trainer_step.py
import torch

def test_one_step_synthetic_backward_no_nan():
    torch.manual_seed(0)
    # Tiny 1D conv model as a proxy for pipeline
    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(1, 8, kernel_size=9, padding=4),
                torch.nn.ReLU(),
                torch.nn.Conv1d(8, 1, kernel_size=9, padding=4),
            )
        def forward(self, x):
            return self.net(x)

    model = TinyDenoiser()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, T = 2, 8000
    clean = torch.randn(B,1,T)*0.05
    noisy = clean + 0.2*torch.randn_like(clean)

    yhat = model(noisy)
    loss = torch.nn.functional.l1_loss(yhat, clean) + 0.1*torch.nn.functional.mse_loss(yhat, clean)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    assert torch.isfinite(gnorm)
    opt.step()
