from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from venom.schedules import append_dims, karras_sigmas


class ConsistencyModel(nn.Module):
    """Consistency training scaffold with EDM-style preconditioning."""

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 28,
        channels: int = 1,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def consistency_function(self, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        sigma_b = append_dims(sigma, x.ndim)
        c_skip = self.sigma_data**2 / (sigma_b**2 + self.sigma_data**2)
        c_out = sigma_b * self.sigma_data / torch.sqrt(sigma_b**2 + self.sigma_data**2)
        c_in = 1 / torch.sqrt(sigma_b**2 + self.sigma_data**2)
        c_noise = torch.log(sigma.clamp(min=1e-20)) / 4
        return c_skip * x + c_out * self.model(c_in * x, c_noise, y)

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x_start.shape[0]
        sigmas = karras_sigmas(40, self.sigma_min, self.sigma_max, device=x_start.device)[:-1].flip(0)
        indices = torch.randint(0, len(sigmas) - 1, (batch_size,), device=x_start.device)
        sigma_t = sigmas[indices]
        sigma_s = sigmas[indices + 1]
        noise = torch.randn_like(x_start)
        x_t = x_start + append_dims(sigma_t, x_start.ndim) * noise
        x_s = x_start + append_dims(sigma_s, x_start.ndim) * noise
        pred_t = self.consistency_function(x_t, sigma_t, y)
        with torch.no_grad():
            pred_s = self.consistency_function(x_s, sigma_s, y)
        weight = append_dims(1 / (sigma_t - sigma_s).abs().clamp(min=1e-3), x_start.ndim)
        return (weight * (pred_t - pred_s).pow(2)).mean()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 1,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigmas = karras_sigmas(max(steps, 1), self.sigma_min, self.sigma_max, device=device)
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device) * sigmas[0]
        for i in range(max(1, steps)):
            sigma = sigmas[i].expand(batch_size)
            x = self.consistency_function(x, sigma, y)
            if i + 1 < len(sigmas) - 1:
                x = x + torch.randn_like(x) * sigmas[i + 1]
        return x.clamp(-1, 1)


class ShortcutModel(nn.Module):
    """Shortcut flow model conditioned on current time and requested step size."""

    def __init__(self, model: nn.Module, image_size: int = 28, channels: int = 1, eps: float = 1e-5):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.eps = eps

    def _cond(self, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return torch.stack([t, torch.log(dt.clamp(min=self.eps))], dim=1)

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x0 = torch.randn_like(x_start)
        x1 = x_start
        dt = 2 ** (-torch.randint(1, 8, (x_start.shape[0],), device=x_start.device).float())
        t = torch.rand_like(dt) * (1 - dt - self.eps)
        t_b = append_dims(t, x_start.ndim)
        x_t = (1 - t_b) * x0 + t_b * x1
        target = x1 - x0
        pred = self.model(x_t, self._cond(t, dt), y)

        mid_t = (t + 0.5 * dt).clamp(max=1 - self.eps)
        with torch.no_grad():
            half_dt = 0.5 * dt
            v1 = self.model(x_t, self._cond(t, half_dt), y)
            x_mid = x_t + append_dims(half_dt, x_start.ndim) * v1
            v2 = self.model(x_mid, self._cond(mid_t, half_dt), y)
            bootstrap = 0.5 * (v1 + v2)
        return F.mse_loss(pred, 0.5 * target + 0.5 * bootstrap)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 1,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        dt = torch.full((batch_size,), 1.0 / max(steps, 1), device=device)
        for i in range(max(steps, 1)):
            t = torch.full((batch_size,), i / max(steps, 1), device=device)
            velocity = self.model(x, self._cond(t, dt), y)
            x = x + append_dims(dt, x.ndim) * velocity
        return x.clamp(-1, 1)


class MeanFlow(nn.Module):
    """MeanFlow-style average velocity training for one-step generation."""

    def __init__(self, model: nn.Module, image_size: int = 28, channels: int = 1, eps: float = 1e-5):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.eps = eps

    def _cond(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.stack([r, t], dim=1)

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x0 = torch.randn_like(x_start)
        x1 = x_start
        r = torch.rand(x_start.shape[0], device=x_start.device) * (1 - self.eps)
        t = r + torch.rand_like(r) * (1 - r)
        r_b = append_dims(r, x_start.ndim)
        t_b = append_dims(t, x_start.ndim)
        x_r = (1 - r_b) * x0 + r_b * x1
        x_t = (1 - t_b) * x0 + t_b * x1
        target_avg_velocity = (x_t - x_r) / append_dims((t - r).clamp(min=self.eps), x_start.ndim)
        pred = self.model(x_r, self._cond(r, t), y)
        return F.mse_loss(pred, target_avg_velocity)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 1,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        if steps <= 1:
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)
            return (x + self.model(x, self._cond(r, t), y)).clamp(-1, 1)

        dt = 1.0 / steps
        for i in range(steps):
            r = torch.full((batch_size,), i * dt, device=device)
            t = torch.full((batch_size,), (i + 1) * dt, device=device)
            x = x + dt * self.model(x, self._cond(r, t), y)
        return x.clamp(-1, 1)


class ProgressiveDistillation(nn.Module):
    """Teacher-student progressive distillation helper for DDPM-family models."""

    def __init__(self, student, teacher, student_steps: int = 50):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.student_steps = student_steps

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if not hasattr(self.student, "q_sample") or not hasattr(self.teacher, "model_predictions"):
            raise TypeError("ProgressiveDistillation expects DDPM-family student and teacher objects")

        batch_size = x_start.shape[0]
        max_t = self.student.timesteps - 1
        timesteps = torch.randint(2, max_t + 1, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.student.q_sample(x_start, timesteps, noise)

        with torch.no_grad():
            _, teacher_x0, _ = self.teacher.model_predictions(x_t, timesteps, y)
        _, student_x0, _ = self.student.model_predictions(x_t, timesteps, y)
        return F.mse_loss(student_x0, teacher_x0)
