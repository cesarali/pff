from typing import List, Tuple
import torch
from torch import nn
import torch.nn as nn
import torch.distributed as dist

class MultiHeadLoss(nn.Module):
    """
    Combines multiple losses with `learnable` or `fixed` weights.
    """
    def __init__(self, weights=None, mode=None, number_of_losses=2):
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            self.weights = nn.Parameter(torch.zeros(number_of_losses))
        elif mode == "fixed":
            self.weights = torch.tensor(weights if weights else torch.ones(number_of_losses))

    def forward(self, losses) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.mode == "learnable":
            combined_loss = sum(
                torch.exp(-self.weights[i]) * losses[i] + self.weights[i]
                for i in range(len(losses))
            )
        elif self.mode == "fixed":
            combined_loss = sum(self.weights[i] * losses[i] for i in range(len(losses)))
        return combined_loss, losses

    def get_weights(self) -> List[float]:
        if self.mode == "learnable":
            return [torch.exp(-weight).item() for weight in self.weights]
        elif self.mode == "fixed":
            return self.weights.tolist()