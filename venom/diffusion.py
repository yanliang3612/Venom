from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from venom.schedules import extract, make_beta_schedule


@dataclass(frozen=True)
class ImprovedDDPMConfig:
    """Convenience preset for cosine schedule + learned range variance."""

    beta_schedule: str = "cosine"
    prediction_type: str = "epsilon"
    variance_type: str = "learned_range"


class GaussianDiffusion(nn.Module):
    """Discrete-time Gaussian diffusion objective used by DDPM-style models.

    Supported variants:
    - DDPM: linear betas, epsilon prediction, fixed variance.
    - Improved DDPM: cosine betas and learned-range variance.
    - ADM / guided-diffusion: class-conditional model, learned variance, optional classifier guidance.
    - Classifier-free guidance: conditional model with null-label dropout plus guidance_scale at sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 28,
        channels: int = 1,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        if prediction_type not in {"epsilon", "x0", "v"}:
            raise ValueError("prediction_type must be one of: epsilon, x0, v")
        if variance_type not in {"fixed_small", "fixed_large", "learned", "learned_range"}:
            raise ValueError("variance_type must be one of: fixed_small, fixed_large, learned, learned_range")

        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.variance_type = variance_type

        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]).clamp(min=1e-20)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("log_one_minus_alpha_cumprod", torch.log((1.0 - alpha_cumprod).clamp(min=1e-20)))
        self.register_buffer("sqrt_recip_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipm1_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod),
        )

    @property
    def learns_variance(self) -> bool:
        return self.variance_type in {"learned", "learned_range"}

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alpha_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alpha_cumprod, timesteps, x_t.shape) * eps
        )

    def predict_eps_from_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alpha_cumprod, timesteps, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alpha_cumprod, timesteps, x_t.shape)

    def predict_x0_from_v(self, x_t: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_t.shape) * v
        )

    def predict_v(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * x_start
        )

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        variance = extract(self.posterior_variance, timesteps, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return mean, variance, log_variance

    def _split_model_output(self, model_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.learns_variance:
            return model_output, None
        return model_output.chunk(2, dim=1)

    def _guided_model_output(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if guidance_scale == 1.0 or y is None:
            return self.model(x, timesteps, y)

        conditional = self.model(x, timesteps, y, force_uncond=False)
        unconditional = self.model(x, timesteps, y, force_uncond=True)
        cond_pred, cond_var = self._split_model_output(conditional)
        uncond_pred, _ = self._split_model_output(unconditional)
        guided_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        if cond_var is None:
            return guided_pred
        return torch.cat([guided_pred, cond_var], dim=1)

    def model_predictions(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        model_output = self._guided_model_output(x, timesteps, y, guidance_scale)
        pred, variance_values = self._split_model_output(model_output)

        if self.prediction_type == "epsilon":
            pred_eps = pred
            pred_x0 = self.predict_x0_from_eps(x, timesteps, pred_eps)
        elif self.prediction_type == "x0":
            pred_x0 = pred
            pred_eps = self.predict_eps_from_x0(x, timesteps, pred_x0)
        else:
            pred_x0 = self.predict_x0_from_v(x, timesteps, pred)
            pred_eps = self.predict_eps_from_x0(x, timesteps, pred_x0)

        if clip_x0:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        return pred_eps, pred_x0, variance_values

    def p_mean_variance(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, pred_x0, variance_values = self.model_predictions(x, timesteps, y, guidance_scale, clip_x0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_x0, x, timesteps)

        if self.variance_type == "fixed_small":
            model_variance = posterior_variance
            model_log_variance = posterior_log_variance
        elif self.variance_type == "fixed_large":
            variance = torch.cat([self.posterior_variance[1:2], self.betas[1:]])
            model_variance = extract(variance, timesteps, x.shape)
            model_log_variance = torch.log(model_variance.clamp(min=1e-20))
        elif self.variance_type == "learned":
            if variance_values is None:
                raise RuntimeError("learned variance requires the model to output 2 * channels")
            model_log_variance = variance_values
            model_variance = torch.exp(model_log_variance)
        else:
            if variance_values is None:
                raise RuntimeError("learned_range variance requires the model to output 2 * channels")
            min_log = extract(self.posterior_log_variance_clipped, timesteps, x.shape)
            max_log = extract(torch.log(self.betas.clamp(min=1e-20)), timesteps, x.shape)
            frac = (variance_values + 1.0) * 0.5
            model_log_variance = frac * max_log + (1.0 - frac) * min_log
            model_variance = torch.exp(model_log_variance)

        return model_mean, model_variance, model_log_variance, pred_x0

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x_start.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps, noise)
        model_output = self.model(x_noisy, timesteps, y)
        pred, _ = self._split_model_output(model_output)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x_start
        else:
            target = self.predict_v(x_start, timesteps, noise)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        classifier_guidance_fn=None,
    ) -> torch.Tensor:
        model_mean, model_variance, model_log_variance, _ = self.p_mean_variance(
            x,
            timesteps,
            y=y,
            guidance_scale=guidance_scale,
        )
        if classifier_guidance_fn is not None and y is not None:
            grad = classifier_guidance_fn(x, timesteps, y)
            model_mean = model_mean + model_variance * grad

        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        classifier_guidance_fn=None,
    ) -> torch.Tensor:
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, y, guidance_scale, classifier_guidance_fn)
        return x.clamp(-1, 1)
