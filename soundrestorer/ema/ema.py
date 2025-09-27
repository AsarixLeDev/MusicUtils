import torch
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=float(decay)
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        d=self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
    def apply_to(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    v.copy_(self.shadow[k])
