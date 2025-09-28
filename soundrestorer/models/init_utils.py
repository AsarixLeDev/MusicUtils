# soundrestorer/models/init_utils.py
import torch, torch.nn as nn

def init_head_for_mask(model: nn.Module, mask_variant: str = "plain") -> bool:
    mv = str(mask_variant).lower()
    head = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 2:
            head = m
    if head is None: return False
    if head.bias is None:
        head.bias = nn.Parameter(torch.zeros(2, device=head.weight.device, dtype=head.weight.dtype))
    with torch.no_grad():
        if mv == "plain":
            head.bias.zero_(); head.bias.data[0] = 1.0  # Re=+1, Im=0
        else:  # delta1, mag, mag_sigm1, mag_delta1
            head.bias.zero_()                           # Re=0, Im=0 (delta1 → 1+Re)
    print(f"[init] head bias set for mask_variant={mv}: bias={head.bias.detach().cpu().tolist()}")
    return True