from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps.long())
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    return torch.cos((t + s) / (1 + s) * math.pi * 0.5).pow(2)


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    max_beta: float = 0.999,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    if schedule == "cosine":
        steps = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        alpha_bar = cosine_alpha_bar(steps)
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(0, max_beta).float()

    if schedule == "quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32).pow(2)

    raise ValueError(f"Unknown beta schedule: {schedule}")


def make_ddim_timesteps(total_steps: int, sample_steps: int, device: torch.device) -> torch.Tensor:
    if sample_steps > total_steps:
        raise ValueError("sample_steps cannot exceed total diffusion steps")
    return torch.linspace(0, total_steps - 1, sample_steps, device=device).long().flip(0)


def betas_for_alpha_bar(timesteps: int, alpha_bar_fn, max_beta: float = 0.999) -> torch.Tensor:
    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    return x.reshape(*x.shape, *((1,) * (target_ndim - x.ndim)))


def karras_sigmas(
    steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    ramp = torch.linspace(0, 1, steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).pow(rho)
    return F.pad(sigmas, (0, 1), value=0.0)
