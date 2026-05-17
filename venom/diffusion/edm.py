from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from venom.diffusion.schedules import append_dims, karras_sigmas


class EDMDiffusion(nn.Module):
    """EDM-style denoising objective and Karras sampler.

    The wrapped model predicts a residual in the preconditioned EDM parameterization.
    It can reuse the same UNet2D architecture as DDPM by passing continuous noise
    levels as the timestep input.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 28,
        channels: int = 1,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std

    def _coefficients(self, sigma: torch.Tensor, ndim: int):
        sigma = append_dims(sigma, ndim)
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        return c_skip, c_out, c_in

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        c_skip, c_out, c_in = self._coefficients(sigma, x.ndim)
        c_noise = torch.log(sigma.clamp(min=1e-20)) / 4
        model_out = self.model(c_in * x, c_noise, y)
        return c_skip * x + c_out * model_out

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        rnd_normal = torch.randn(x_start.shape[0], device=x_start.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(x_start) * append_dims(sigma, x_start.ndim)
        denoised = self.denoise(x_start + noise, sigma, y)
        weight = append_dims((sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2, x_start.ndim)
        return (weight * (denoised - x_start).pow(2)).mean()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 32,
        y: torch.Tensor | None = None,
        heun: bool = True,
    ) -> torch.Tensor:
        sigmas = karras_sigmas(steps, self.sigma_min, self.sigma_max, device=device)
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device) * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].expand(batch_size)
            sigma_next = sigmas[i + 1].expand(batch_size)
            denoised = self.denoise(x, sigma, y)
            d = (x - denoised) / append_dims(sigma, x.ndim)
            dt = append_dims(sigma_next - sigma, x.ndim)
            x_euler = x + d * dt

            if heun and sigmas[i + 1] > 0:
                denoised_next = self.denoise(x_euler, sigma_next, y)
                d_next = (x_euler - denoised_next) / append_dims(sigma_next, x.ndim)
                x = x + 0.5 * (d + d_next) * dt
            else:
                x = x_euler

        return x.clamp(-1, 1)
