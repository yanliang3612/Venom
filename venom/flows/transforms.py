from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 512, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current, hidden_dim), nn.ReLU(inplace=True)])
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class FlowTransform(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ActNorm(FlowTransform):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(1e-4)
        self.bias.data.copy_(-mean)
        self.log_scale.data.copy_(-std.log())
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.initialized):
            self.initialize(x)
        z = (x + self.bias) * torch.exp(self.log_scale)
        logdet = self.log_scale.sum().expand(x.shape[0])
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * torch.exp(-self.log_scale) - self.bias


class InvertibleLinear(FlowTransform):
    def __init__(self, dim: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.weight = nn.Parameter(q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x @ self.weight.t()
        _, logabsdet = torch.linalg.slogdet(self.weight)
        return z, logabsdet.expand(x.shape[0])

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z @ torch.linalg.inv(self.weight).t()


class AdditiveCoupling(FlowTransform):
    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 512):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = mlp(dim, dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        shift = self.net(x_masked) * (1.0 - self.mask)
        return x_masked + (1.0 - self.mask) * (x + shift), x.new_zeros(x.shape[0])

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z_masked = z * self.mask
        shift = self.net(z_masked) * (1.0 - self.mask)
        return z_masked + (1.0 - self.mask) * (z - shift)


class AffineCoupling(FlowTransform):
    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 512, scale: float = 2.0):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = mlp(dim, dim * 2, hidden_dim=hidden_dim)
        self.scale = scale

    def _params(self, x_masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(x_masked).chunk(2, dim=1)
        inv_mask = 1.0 - self.mask
        return shift * inv_mask, torch.tanh(log_scale) * self.scale * inv_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        shift, log_scale = self._params(x_masked)
        z = x_masked + (1.0 - self.mask) * ((x + shift) * torch.exp(log_scale))
        return z, log_scale.sum(dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z_masked = z * self.mask
        shift, log_scale = self._params(z_masked)
        return z_masked + (1.0 - self.mask) * (z * torch.exp(-log_scale) - shift)


class SplineCoupling(FlowTransform):
    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 512, num_terms: int = 4):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.num_terms = num_terms
        self.net = mlp(dim, dim * num_terms * 3, hidden_dim=hidden_dim)

    def _params(self, x_masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.net(x_masked).reshape(x_masked.shape[0], x_masked.shape[1], self.num_terms, 3)
        a, b, c = params.unbind(dim=-1)
        a = 0.05 * F.softplus(a)
        b = F.softplus(b) + 1e-3
        c = torch.tanh(c)
        inv_mask = (1.0 - self.mask).view(1, -1, 1)
        return a * inv_mask, b * inv_mask, c * inv_mask

    def _warp(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_terms = x.unsqueeze(-1)
        h = torch.tanh(b * x_terms + c)
        y = x + (a * h).sum(dim=-1)
        derivative = 1.0 + (a * b * (1.0 - h.pow(2))).sum(dim=-1)
        return y, derivative.clamp_min(1e-4).log()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        a, b, c = self._params(x_masked)
        warped, logdet = self._warp(x, a, b, c)
        z = x_masked + (1.0 - self.mask) * warped
        return z, (logdet * (1.0 - self.mask)).sum(dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z_masked = z * self.mask
        a, b, c = self._params(z_masked)
        x = z
        for _ in range(12):
            warped, _ = self._warp(x, a, b, c)
            x = z_masked + (1.0 - self.mask) * (x + z - warped)
        return x


class PlanarTransform(FlowTransform):
    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def _u_hat(self) -> torch.Tensor:
        wu = (self.w * self.u).sum()
        m = -1.0 + F.softplus(wu)
        return self.u + (m - wu) * self.w / self.w.pow(2).sum().clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat = self._u_hat()
        inner = x @ self.w + self.b
        h = torch.tanh(inner)
        z = x + h.unsqueeze(1) * u_hat
        psi = (1.0 - h.pow(2)).unsqueeze(1) * self.w
        logdet = torch.log(torch.abs(1.0 + psi @ u_hat.unsqueeze(1)).squeeze(1).clamp_min(1e-6))
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        u_hat = self._u_hat()
        for _ in range(30):
            x = z - torch.tanh(x @ self.w + self.b).unsqueeze(1) * u_hat
        return x


class RadialTransform(FlowTransform):
    def __init__(self, dim: int):
        super().__init__()
        self.z0 = nn.Parameter(torch.zeros(dim))
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.beta_unconstrained = nn.Parameter(torch.zeros(()))

    def _params(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.log_alpha) + 1e-4
        beta = -alpha + F.softplus(self.beta_unconstrained)
        return alpha, beta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, beta = self._params()
        diff = x - self.z0
        r = diff.norm(dim=1)
        h = 1.0 / (alpha + r)
        z = x + beta * h.unsqueeze(1) * diff
        hp = -1.0 / (alpha + r).pow(2)
        d = x.shape[1]
        logdet = (d - 1) * torch.log1p(beta * h) + torch.log1p(beta * h + beta * hp * r)
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        alpha, beta = self._params()
        for _ in range(30):
            diff = x - self.z0
            h = 1.0 / (alpha + diff.norm(dim=1))
            x = z - beta * h.unsqueeze(1) * diff
        return x


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask.copy_(mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512, output_multiplier: int = 2):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            MaskedLinear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            MaskedLinear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            MaskedLinear(hidden_dim, dim * output_multiplier),
        )
        input_degrees = torch.arange(1, dim + 1)
        hidden_degrees = torch.arange(hidden_dim) % max(dim - 1, 1) + 1
        output_degrees = torch.arange(1, dim + 1).repeat(output_multiplier)
        layers = [module for module in self.net if isinstance(module, MaskedLinear)]
        layers[0].set_mask((hidden_degrees[:, None] >= input_degrees[None, :]).float())
        layers[1].set_mask((hidden_degrees[:, None] >= hidden_degrees[None, :]).float())
        layers[2].set_mask((output_degrees[:, None] > hidden_degrees[None, :]).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedAutoregressiveTransform(FlowTransform):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        self.net = MADE(dim, hidden_dim=hidden_dim, output_multiplier=2)

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(x).chunk(2, dim=1)
        return shift, torch.tanh(log_scale).clamp(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self._params(x)
        z = (x - shift) * torch.exp(-log_scale)
        return z, -log_scale.sum(dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(z)
        for index in range(self.dim):
            shift, log_scale = self._params(x)
            x[:, index] = z[:, index] * torch.exp(log_scale[:, index]) + shift[:, index]
        return x


class InverseAutoregressiveTransform(FlowTransform):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        self.net = MADE(dim, hidden_dim=hidden_dim, output_multiplier=2)

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(x).chunk(2, dim=1)
        return shift, torch.tanh(log_scale).clamp(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self._params(x)
        z = x * torch.exp(log_scale) + shift
        return z, log_scale.sum(dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(z)
        for index in range(self.dim):
            shift, log_scale = self._params(x)
            x[:, index] = (z[:, index] - shift[:, index]) * torch.exp(-log_scale[:, index])
        return x


class CNFTransform(FlowTransform):
    def __init__(self, dim: int, hidden_dim: int = 256, steps: int = 8):
        super().__init__()
        self.steps = steps
        self.net = mlp(dim + 1, dim, hidden_dim=hidden_dim)

    def _velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        t_column = x.new_full((x.shape[0], 1), t)
        return self.net(torch.cat([x, t_column], dim=1))

    def _trace_estimate(self, x: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        probe = torch.empty_like(x).bernoulli_(0.5) * 2.0 - 1.0
        jvp = torch.autograd.grad((velocity * probe).sum(), x, create_graph=True, retain_graph=True)[0]
        return (jvp * probe).sum(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = x.new_zeros(x.shape[0])
        dt = 1.0 / self.steps
        for index in range(self.steps):
            t = index / self.steps
            z = z.requires_grad_(True)
            velocity = self._velocity(z, t)
            trace = self._trace_estimate(z, velocity)
            z = (z + dt * velocity).detach() + dt * velocity - dt * velocity.detach()
            logdet = logdet + dt * trace
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        dt = 1.0 / self.steps
        for index in reversed(range(self.steps)):
            t = index / self.steps
            x = x - dt * self._velocity(x, t)
        return x


def alternating_mask(dim: int, parity: int = 0) -> torch.Tensor:
    return ((torch.arange(dim) + parity) % 2).float()
