from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from venom.diffusion.schedules import append_dims


class FlowMatchingDiffusion(nn.Module):
    """Continuous-time velocity-field models.

    The class covers rectified flow, flow matching, conditional flow matching,
    lightweight minibatch OT-CFM, and stochastic interpolants with a shared
    model API: the network predicts v_t(x).
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 28,
        channels: int = 1,
        variant: str = "rectified-flow",
        path_sigma: float = 0.0,
        interpolant_gamma: float = 0.25,
        eps: float = 1e-5,
    ):
        super().__init__()
        if variant not in {
            "rectified-flow",
            "flow-matching",
            "conditional-flow-matching",
            "ot-cfm",
            "stochastic-interpolants",
        }:
            raise ValueError(f"Unknown flow matching variant: {variant}")
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.variant = variant
        self.path_sigma = path_sigma
        self.interpolant_gamma = interpolant_gamma
        self.eps = eps

    def _source(self, x_target: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x_target)

    def _approximate_ot_coupling(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Pair source and target minibatches by a random-projection OT surrogate.

        Exact minibatch OT usually needs Hungarian/Sinkhorn solvers. This keeps
        the package dependency-light while still shortening average paths versus
        arbitrary pairings.
        """

        batch = x_source.shape[0]
        flat_source = x_source.reshape(batch, -1)
        flat_target = x_target.reshape(batch, -1)
        projection = torch.randn(flat_source.shape[1], device=x_source.device, dtype=x_source.dtype)
        source_order = torch.argsort(flat_source @ projection)
        target_order = torch.argsort(flat_target @ projection)
        paired = torch.empty_like(x_source)
        paired[target_order] = x_source[source_order]
        return paired

    def _path(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_b = append_dims(t, x_target.ndim)
        if self.variant == "stochastic-interpolants":
            z = torch.randn_like(x_target)
            gamma = self.interpolant_gamma * torch.sin(math.pi * t)
            gamma_prime = self.interpolant_gamma * math.pi * torch.cos(math.pi * t)
            x_t = (1 - t_b) * x_source + t_b * x_target + append_dims(gamma, x_target.ndim) * z
            velocity = x_target - x_source + append_dims(gamma_prime, x_target.ndim) * z
            return x_t, velocity

        x_t = (1 - t_b) * x_source + t_b * x_target
        velocity = x_target - x_source
        if self.path_sigma > 0:
            noise = torch.randn_like(x_target)
            sigma_t = self.path_sigma * torch.sin(math.pi * t).clamp(min=0)
            sigma_prime = self.path_sigma * math.pi * torch.cos(math.pi * t)
            x_t = x_t + append_dims(sigma_t, x_target.ndim) * noise
            velocity = velocity + append_dims(sigma_prime, x_target.ndim) * noise
        return x_t, velocity

    def training_loss(self, x_start: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x_target = x_start
        x_source = self._source(x_target)
        if self.variant == "ot-cfm":
            x_source = self._approximate_ot_coupling(x_source, x_target)

        t = torch.rand(x_target.shape[0], device=x_target.device) * (1 - 2 * self.eps) + self.eps
        x_t, target_velocity = self._path(x_source, x_target, t)
        pred_velocity = self.model(x_t, t, y)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 50,
        y: torch.Tensor | None = None,
        method: str = "heun",
    ) -> torch.Tensor:
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")

        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        times = torch.linspace(0.0, 1.0, steps + 1, device=device)
        for i in range(steps):
            t = times[i].expand(batch_size)
            t_next = times[i + 1].expand(batch_size)
            dt = times[i + 1] - times[i]
            velocity = self.model(x, t, y)
            if method == "euler":
                x = x + dt * velocity
            else:
                x_euler = x + dt * velocity
                velocity_next = self.model(x_euler, t_next, y)
                x = x + 0.5 * dt * (velocity + velocity_next)
        return x.clamp(-1, 1)


class RectifiedFlow(FlowMatchingDiffusion):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, variant="rectified-flow", **kwargs)


class FlowMatching(FlowMatchingDiffusion):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, variant="flow-matching", **kwargs)


class ConditionalFlowMatching(FlowMatchingDiffusion):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, variant="conditional-flow-matching", **kwargs)


class OptimalTransportCFM(FlowMatchingDiffusion):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, variant="ot-cfm", **kwargs)


class StochasticInterpolants(FlowMatchingDiffusion):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, variant="stochastic-interpolants", **kwargs)
