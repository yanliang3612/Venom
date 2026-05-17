from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for group_count in (32, 16, 8, 4, 2, 1):
        if channels % group_count == 0:
            return group_count
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim > 1:
            pieces = []
            for index in range(timesteps.shape[1]):
                pieces.append((index + 1) * self.forward(timesteps[:, index]))
            return torch.stack(pieces, dim=0).sum(dim=0) / timesteps.shape[1]

        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        head_dim = channels // self.num_heads
        q = q.reshape(batch * self.num_heads, head_dim, height * width).transpose(1, 2)
        k = k.reshape(batch * self.num_heads, head_dim, height * width)
        v = v.reshape(batch * self.num_heads, head_dim, height * width).transpose(1, 2)

        weight = torch.bmm(q, k) * (head_dim**-0.5)
        weight = weight.softmax(dim=-1)
        h = torch.bmm(weight, v).transpose(1, 2).reshape(batch, channels, height * width)
        h = self.proj(h).reshape(batch, channels, height, width)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """Compact ADM-style U-Net for MNIST-sized images.

    It supports class conditioning, classifier-free dropout, attention at the
    bottleneck, and doubled output channels for learned-variance diffusion.
    """

    def __init__(
        self,
        image_channels: int = 1,
        out_channels: int | None = None,
        base_channels: int = 64,
        time_dim: int = 256,
        num_classes: int | None = None,
        class_dropout: float = 0.0,
        dropout: float = 0.0,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.out_channels = out_channels or image_channels
        self.num_classes = num_classes
        self.class_dropout = class_dropout
        self.null_class = num_classes if num_classes is not None else None

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.label_emb = (
            nn.Embedding(num_classes + 1, time_dim)
            if num_classes is not None
            else None
        )

        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim, dropout)
        self.downsample1 = Downsample(base_channels)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim, dropout)
        self.downsample2 = Downsample(base_channels * 2)

        self.mid1 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim, dropout)
        self.attn = AttentionBlock(base_channels * 4, num_heads=attention_heads)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim, dropout)

        self.upsample2 = Upsample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, time_dim, dropout)
        self.upsample1 = Upsample(base_channels)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_dim, dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.out_channels, 3, padding=1),
        )

    def _label_embedding(
        self,
        y: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        force_uncond: bool = False,
    ) -> torch.Tensor | None:
        if self.label_emb is None:
            return None

        if y is None or force_uncond:
            labels = torch.full((batch_size,), self.null_class, device=device, dtype=torch.long)
        else:
            labels = y.to(device=device, dtype=torch.long)
            if self.training and self.class_dropout > 0:
                drop = torch.rand(batch_size, device=device) < self.class_dropout
                labels = torch.where(drop, torch.full_like(labels, self.null_class), labels)
        return self.label_emb(labels)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        label_emb = self._label_embedding(y, x.shape[0], x.device, force_uncond)
        if label_emb is not None:
            time_emb = time_emb + label_emb

        x = self.init_conv(x)
        skip1 = self.down1(x, time_emb)
        x = self.downsample1(skip1)
        skip2 = self.down2(x, time_emb)
        x = self.downsample2(skip2)

        x = self.mid1(x, time_emb)
        x = self.attn(x)
        x = self.mid2(x, time_emb)

        x = self.upsample2(x)
        x = self.up2(torch.cat([x, skip2], dim=1), time_emb)
        x = self.upsample1(x)
        x = self.up1(torch.cat([x, skip1], dim=1), time_emb)
        return self.out(x)
