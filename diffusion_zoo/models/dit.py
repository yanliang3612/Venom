from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_zoo.models.unet import SinusoidalTimeEmbedding


class PatchEmbed(nn.Module):
    def __init__(self, image_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(image_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.cond(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_msa[:, None] * attn_out
        h = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.mlp(h)
        return x


class DiT(nn.Module):
    """Small DiT-style backbone for MNIST.

    This follows the DiT idea of patch tokens plus adaptive layer norm
    conditioning, scaled down for 28x28 experiments.
    """

    def __init__(
        self,
        image_size: int = 28,
        image_channels: int = 1,
        out_channels: int | None = None,
        patch_size: int = 4,
        hidden_size: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        time_dim: int | None = None,
        num_classes: int | None = None,
        class_dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.image_channels = image_channels
        self.out_channels = out_channels or image_channels
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.num_classes = num_classes
        self.class_dropout = class_dropout
        self.null_class = num_classes if num_classes is not None else None

        time_dim = time_dim or hidden_size
        self.patch_embed = PatchEmbed(image_channels, hidden_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.label_emb = nn.Embedding(num_classes + 1, hidden_size) if num_classes is not None else None
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_cond = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

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

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        p = self.patch_size
        c = self.out_channels
        h = w = self.grid_size
        x = x.reshape(batch, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(batch, c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        tokens = self.patch_embed(x) + self.pos_embed
        cond = self.time_mlp(timesteps)
        label_emb = self._label_embedding(y, x.shape[0], x.device, force_uncond)
        if label_emb is not None:
            cond = cond + label_emb

        for block in self.blocks:
            tokens = block(tokens, cond)

        shift, scale = self.final_cond(cond).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        patches = self.final_linear(tokens)
        return self.unpatchify(patches)
