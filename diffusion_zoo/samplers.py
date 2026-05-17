from __future__ import annotations

import torch

from diffusion_zoo.schedules import make_ddim_timesteps


class DDPMSampler:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, **kwargs) -> torch.Tensor:
        return self.diffusion.sample(batch_size=batch_size, device=device, **kwargs)


class DDIMSampler:
    def __init__(self, diffusion, steps: int = 50, eta: float = 0.0):
        self.diffusion = diffusion
        self.steps = steps
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        d = self.diffusion
        x = torch.randn(batch_size, d.channels, d.image_size, d.image_size, device=device)
        schedule = make_ddim_timesteps(d.timesteps, self.steps, device)

        for i, step in enumerate(schedule):
            timesteps = torch.full((batch_size,), step.item(), device=device, dtype=torch.long)
            pred_eps, pred_x0, _ = d.model_predictions(x, timesteps, y, guidance_scale)
            next_step = schedule[i + 1].item() if i + 1 < len(schedule) else -1
            alpha = d.alpha_cumprod[step]
            alpha_next = d.alpha_cumprod[next_step] if next_step >= 0 else torch.tensor(1.0, device=device)

            sigma = self.eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
            direction = torch.sqrt((1 - alpha_next - sigma**2).clamp(min=0.0)) * pred_eps
            noise = sigma * torch.randn_like(x) if next_step >= 0 else 0
            x = torch.sqrt(alpha_next) * pred_x0 + direction + noise

        return x.clamp(-1, 1)


class DPMSolverSampler:
    """Lightweight first-order DPM-Solver and DPM-Solver++ sampler.

    algorithm="dpmsolver" uses noise prediction. algorithm="dpmsolver++" uses
    data prediction, which is usually better for guided sampling.
    """

    def __init__(self, diffusion, steps: int = 20, algorithm: str = "dpmsolver++"):
        if algorithm not in {"dpmsolver", "dpmsolver++"}:
            raise ValueError("algorithm must be 'dpmsolver' or 'dpmsolver++'")
        self.diffusion = diffusion
        self.steps = steps
        self.algorithm = algorithm

    def _lambda(self, step: torch.Tensor) -> torch.Tensor:
        d = self.diffusion
        alpha = torch.sqrt(d.alpha_cumprod[step])
        sigma = torch.sqrt(1 - d.alpha_cumprod[step])
        return torch.log(alpha.clamp(min=1e-20)) - torch.log(sigma.clamp(min=1e-20))

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        d = self.diffusion
        x = torch.randn(batch_size, d.channels, d.image_size, d.image_size, device=device)
        schedule = make_ddim_timesteps(d.timesteps, self.steps, device)

        for i, step in enumerate(schedule):
            timesteps = torch.full((batch_size,), step.item(), device=device, dtype=torch.long)
            pred_eps, pred_x0, _ = d.model_predictions(x, timesteps, y, guidance_scale)

            if i + 1 == len(schedule):
                x = pred_x0
                break

            next_step = schedule[i + 1]
            alpha_s = torch.sqrt(d.alpha_cumprod[step])
            sigma_s = torch.sqrt(1 - d.alpha_cumprod[step])
            alpha_t = torch.sqrt(d.alpha_cumprod[next_step])
            sigma_t = torch.sqrt(1 - d.alpha_cumprod[next_step])
            h = self._lambda(next_step) - self._lambda(step)

            if self.algorithm == "dpmsolver":
                x = (alpha_t / alpha_s) * x - sigma_t * torch.expm1(h) * pred_eps
            else:
                x = (sigma_t / sigma_s) * x + alpha_t * (1 - torch.exp(-h)) * pred_x0

            if not torch.isfinite(x).all():
                raise FloatingPointError("DPM-Solver produced non-finite values")

        return x.clamp(-1, 1)


def make_sampler(diffusion, name: str, steps: int = 50, eta: float = 0.0):
    normalized = name.lower()
    if normalized == "ddpm":
        return DDPMSampler(diffusion)
    if normalized == "ddim":
        return DDIMSampler(diffusion, steps=steps, eta=eta)
    if normalized == "dpm-solver":
        return DPMSolverSampler(diffusion, steps=steps, algorithm="dpmsolver")
    if normalized in {"dpm-solver++", "dpmsolver++"}:
        return DPMSolverSampler(diffusion, steps=steps, algorithm="dpmsolver++")
    raise ValueError(f"Unknown sampler: {name}")
