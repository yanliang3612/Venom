from __future__ import annotations

import torch

from venom.edm import EDMDiffusion
from venom.schedules import append_dims, karras_sigmas


class PFGMPlusPlusDiffusion(EDMDiffusion):
    """PFGM++ training scaffold using the r = sigma * sqrt(D) transfer rule.

    D=1 corresponds to the original PFGM end of the family. Large D approaches
    the diffusion/EDM perturbation kernel.
    """

    def __init__(self, *args, augmented_dim: float = 128.0, data_dim: int = 28 * 28, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmented_dim = float(augmented_dim)
        self.data_dim = int(data_dim)

    def _pfgm_noise(self, shape: torch.Size, sigma: torch.Tensor) -> torch.Tensor:
        batch = shape[0]
        device = sigma.device
        dtype = sigma.dtype
        beta = torch.distributions.Beta(
            torch.tensor(self.data_dim / 2, device=device, dtype=dtype),
            torch.tensor(self.augmented_dim / 2, device=device, dtype=dtype),
        )
        samples = beta.sample((batch,)).clamp(1e-6, 1 - 1e-6)
        inverse_beta = samples / (1 - samples)
        radius = sigma * (self.augmented_dim**0.5) * torch.sqrt(inverse_beta)
        direction = torch.randn(batch, self.data_dim, device=device, dtype=dtype)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return (direction * radius[:, None]).reshape(shape)

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        rnd_normal = torch.randn(x_start.shape[0], device=x_start.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        noise = self._pfgm_noise(x_start.shape, sigma).to(x_start.dtype)
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
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = self._pfgm_noise(torch.Size(shape), sigmas[0].expand(batch_size)).to(device)

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


class PFGMDiffusion(PFGMPlusPlusDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, augmented_dim=1.0, **kwargs)
