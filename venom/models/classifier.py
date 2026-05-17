from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from venom.models.unet import SinusoidalTimeEmbedding, _groups


class MNISTClassifier(nn.Module):
    """Small timestep-conditioned classifier for ADM-style classifier guidance."""

    def __init__(
        self,
        image_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, base_channels * 4),
        )
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 3, padding=1),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(base_channels * 2), base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.GroupNorm(_groups(base_channels * 4), base_channels * 4),
            nn.SiLU(),
        )
        self.out = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        time_bias = self.time_mlp(timesteps)[:, :, None, None]
        h = F.silu(h + time_bias)
        h = h.mean(dim=(2, 3))
        return self.out(h)
