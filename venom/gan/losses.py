from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_loss(kind: str, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    if kind == "vanilla":
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return real_loss + fake_loss
    if kind == "lsgan":
        return 0.5 * ((real_logits - 1).pow(2).mean() + fake_logits.pow(2).mean())
    if kind == "wgan":
        return fake_logits.mean() - real_logits.mean()
    if kind == "hinge":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    raise ValueError(f"Unknown GAN loss kind: {kind}")


def generator_loss(kind: str, fake_logits: torch.Tensor) -> torch.Tensor:
    if kind == "vanilla":
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if kind == "lsgan":
        return 0.5 * (fake_logits - 1).pow(2).mean()
    if kind in {"wgan", "hinge"}:
        return -fake_logits.mean()
    raise ValueError(f"Unknown GAN loss kind: {kind}")


def gradient_penalty(discriminator, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    batch_size = real.shape[0]
    eps = torch.rand(batch_size, *((1,) * (real.ndim - 1)), device=real.device)
    mixed = eps * real + (1 - eps) * fake.detach()
    mixed.requires_grad_(True)
    mixed_logits = discriminator(mixed, y)["logits"]
    grad = torch.autograd.grad(
        outputs=mixed_logits.sum(),
        inputs=mixed,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (grad.flatten(1).norm(2, dim=1) - 1).pow(2).mean()
