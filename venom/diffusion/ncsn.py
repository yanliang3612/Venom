from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from venom.diffusion.schedules import append_dims


class NCSNDiffusion(nn.Module):
    """Noise Conditional Score Network objective and annealed Langevin sampler."""

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 28,
        channels: int = 1,
        num_sigmas: int = 10,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        version: str = "ncsn",
    ):
        super().__init__()
        if version not in {"ncsn", "ncsnv2"}:
            raise ValueError("version must be 'ncsn' or 'ncsnv2'")
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.version = version
        sigmas = torch.exp(torch.linspace(torch.log(torch.tensor(sigma_max)), torch.log(torch.tensor(sigma_min)), num_sigmas))
        self.register_buffer("sigmas", sigmas.float())

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        labels = torch.randint(0, len(self.sigmas), (x_start.shape[0],), device=x_start.device)
        sigma = self.sigmas[labels]
        noise = torch.randn_like(x_start)
        x_noisy = x_start + append_dims(sigma, x_start.ndim) * noise
        score = self.model(x_noisy, sigma, y)
        loss = (score * append_dims(sigma, x_start.ndim) + noise).pow(2)
        if self.version == "ncsnv2":
            loss = loss * append_dims(sigma.square(), x_start.ndim)
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps_each: int = 10,
        step_lr: float = 2e-5,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.rand(batch_size, self.channels, self.image_size, self.image_size, device=device) * 2 - 1
        for sigma in self.sigmas:
            sigma_batch = sigma.expand(batch_size).to(device)
            step_size = step_lr * (sigma / self.sigmas[-1]).pow(2)
            for _ in range(steps_each):
                grad = self.model(x, sigma_batch, y)
                noise = torch.randn_like(x)
                x = x + step_size * grad + torch.sqrt(2 * step_size) * noise
        return x.clamp(-1, 1)
