from __future__ import annotations

import torch


def energy_sum(model, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    return model.energy(x, y).sum()


def langevin_sample(
    model,
    x: torch.Tensor,
    steps: int = 60,
    step_size: float = 10.0,
    noise_scale: float = 0.005,
    y: torch.Tensor | None = None,
    clamp: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    sample = x.detach()
    with torch.enable_grad():
        for _ in range(steps):
            sample.requires_grad_(True)
            energy = energy_sum(model, sample, y)
            grad = torch.autograd.grad(energy, sample)[0]
            sample = sample - 0.5 * step_size * grad
            if noise_scale > 0:
                sample = sample + noise_scale * torch.randn_like(sample)
            sample = sample.detach().clamp(*clamp)
    model.train(was_training)
    return sample


def sgld_sample(
    model,
    x: torch.Tensor,
    steps: int = 60,
    step_size: float = 1.0,
    noise_scale: float | None = None,
    y: torch.Tensor | None = None,
    clamp: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    if noise_scale is None:
        noise_scale = step_size**0.5
    return langevin_sample(model, x, steps=steps, step_size=step_size, noise_scale=noise_scale, y=y, clamp=clamp)
