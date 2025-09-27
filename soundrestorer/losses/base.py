from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch

class LossModule(ABC):
    @abstractmethod
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Any) -> torch.Tensor:
        ...

    def __call__(self, outputs, batch):
        return self.forward(outputs, batch)
