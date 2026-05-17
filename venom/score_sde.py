from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from venom.schedules import append_dims


class SDE:
    T = 1.0

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class VPSDE(SDE):
    beta_min: float = 0.1
    beta_max: float = 20.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta(t)
        drift = -0.5 * append_dims(beta_t, x.ndim) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = -0.25 * t.square() * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = append_dims(torch.exp(log_mean_coeff), x.ndim) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff).clamp(max=0.999999))
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)


@dataclass
class SubVPSDE(VPSDE):
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta(t)
        discount = 1.0 - torch.exp(-2 * self.beta_min * t - (self.beta_max - self.beta_min) * t.square())
        drift = -0.5 * append_dims(beta_t, x.ndim) * x
        diffusion = torch.sqrt(beta_t * discount.clamp(min=1e-5))
        return drift, diffusion


@dataclass
class VESDE(SDE):
    sigma_min: float = 0.01
    sigma_max: float = 50.0

    @property
    def log_ratio(self) -> float:
        return torch.log(torch.tensor(self.sigma_max / self.sigma_min)).item()

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma(t)
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * self.log_ratio, device=t.device))
        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, self.sigma(t)

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device) * self.sigma_max


class ScoreSDEDiffusion(nn.Module):
    """Continuous-time score matching with Euler predictor-corrector sampling."""

    def __init__(
        self,
        model: nn.Module,
        sde: SDE,
        image_size: int = 28,
        channels: int = 1,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.sde = sde
        self.image_size = image_size
        self.channels = channels
        self.eps = eps

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        t = torch.rand(x_start.shape[0], device=x_start.device) * (self.sde.T - self.eps) + self.eps
        noise = torch.randn_like(x_start)
        mean, std = self.sde.marginal_prob(x_start, t)
        x_noisy = mean + append_dims(std, x_start.ndim) * noise
        score = self.model(x_noisy, t, y)
        return (score * append_dims(std, x_start.ndim) + noise).pow(2).mean()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 500,
        corrector_steps: int = 1,
        snr: float = 0.16,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = self.sde.prior_sampling(shape, device=device)
        times = torch.linspace(self.sde.T, self.eps, steps, device=device)

        for i, time in enumerate(times):
            t = time.expand(batch_size)
            for _ in range(corrector_steps):
                grad = self.model(x, t, y)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean().clamp(min=1e-12)
                noise_norm = torch.norm(noise.reshape(batch_size, -1), dim=-1).mean()
                step_size = (snr * noise_norm / grad_norm).square() * 2
                x = x + step_size * grad + torch.sqrt(2 * step_size) * noise

            drift, diffusion = self.sde.sde(x, t)
            score = self.model(x, t, y)
            reverse_drift = drift - append_dims(diffusion.square(), x.ndim) * score
            if i == len(times) - 1:
                break
            dt = times[i + 1] - time
            x = x + reverse_drift * dt + append_dims(diffusion * torch.sqrt(-dt), x.ndim) * torch.randn_like(x)

        return x.clamp(-1, 1)
