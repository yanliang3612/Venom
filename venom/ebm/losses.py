from __future__ import annotations

import torch
import torch.nn.functional as F


def cd_loss(model, data: torch.Tensor, steps: int = 1, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        negative = model.gibbs(data, steps=steps, y=y)
    positive_energy = model.free_energy(data, y).mean()
    negative_energy = model.free_energy(negative.detach(), y).mean()
    return positive_energy - negative_energy, negative.detach()


def pcd_loss(
    model,
    data: torch.Tensor,
    persistent: torch.Tensor,
    steps: int = 1,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        negative = model.gibbs(persistent, steps=steps, y=y)
    positive_energy = model.free_energy(data, y).mean()
    negative_energy = model.free_energy(negative.detach(), y).mean()
    return positive_energy - negative_energy, negative.detach()


def contrastive_energy_loss(model, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    return model.energy(real, y).mean() - model.energy(fake.detach(), y).mean()


def denoising_score_matching_loss(model, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    perturbed = (x + noise).detach().requires_grad_(True)
    energy = model.energy(perturbed).sum()
    score = -torch.autograd.grad(energy, perturbed, create_graph=True)[0]
    target = -noise / (sigma**2)
    return 0.5 * (score - target).flatten(1).pow(2).sum(dim=1).mean()


def sliced_score_matching_loss(model, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    v = torch.randn_like(x)
    v = v / (v.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
    energy = model.energy(x).sum()
    score = -torch.autograd.grad(energy, x, create_graph=True)[0]
    score_dot_v = (score * v).flatten(1).sum(dim=1)
    grad_score_dot_v = torch.autograd.grad(score_dot_v.sum(), x, create_graph=True)[0]
    trace_estimate = (grad_score_dot_v * v).flatten(1).sum(dim=1)
    quadratic = 0.5 * score_dot_v.pow(2)
    return (quadratic + trace_estimate).mean()


def nce_loss(model, data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    data_logits = -model.energy(data)
    noise_logits = -model.energy(noise)
    data_loss = F.binary_cross_entropy_with_logits(data_logits, torch.ones_like(data_logits))
    noise_loss = F.binary_cross_entropy_with_logits(noise_logits, torch.zeros_like(noise_logits))
    return data_loss + noise_loss
