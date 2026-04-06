from __future__ import annotations

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Parameter-free attention module described in SimAM."""

    def __init__(self, channels: int | None = None, lambda_val: float = 1e-4) -> None:
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = max(x.shape[2] * x.shape[3] - 1, 1)
        centered_sq = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        variance = centered_sq.sum(dim=(2, 3), keepdim=True) / n
        energy_inv = centered_sq / (4 * (variance + self.lambda_val)) + 0.5
        return x * torch.sigmoid(energy_inv)


__all__ = ("SimAM",)


