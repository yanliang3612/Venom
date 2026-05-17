from __future__ import annotations

import torch
import torch.nn as nn

from venom.flows.dequantization import bits_per_dim, gaussian_log_prob, inverse_logit_preprocess, logit_preprocess
from venom.flows.transforms import (
    ActNorm,
    AdditiveCoupling,
    AffineCoupling,
    CNFTransform,
    InverseAutoregressiveTransform,
    InvertibleLinear,
    MaskedAutoregressiveTransform,
    PlanarTransform,
    RadialTransform,
    SplineCoupling,
    alternating_mask,
)


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        transforms: list[nn.Module],
        image_size: int = 28,
        channels: int = 1,
        alpha: float = 1e-5,
        dequantize: bool = True,
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.image_size = image_size
        self.channels = channels
        self.dim = image_size * image_size * channels
        self.alpha = alpha
        self.dequantize = dequantize

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, logdet = logit_preprocess(images, alpha=self.alpha, dequantize=self.dequantize)
        for transform in self.transforms:
            z, inc = transform(z)
            logdet = logdet + inc
        return z, logdet

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return inverse_logit_preprocess(x, image_size=self.image_size, channels=self.channels, alpha=self.alpha)

    def log_prob(self, images: torch.Tensor) -> torch.Tensor:
        z, logdet = self.encode(images)
        return gaussian_log_prob(z) + logdet

    def training_loss(self, images: torch.Tensor) -> torch.Tensor:
        nll = -self.log_prob(images)
        return bits_per_dim(nll, self.dim).mean()

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.dim, device=device)
        return self.decode(z)


class NICE(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 6, hidden_dim: int = 512):
        transforms = [
            AdditiveCoupling(dim, alternating_mask(dim, layer), hidden_dim=hidden_dim)
            for layer in range(num_layers)
        ]
        super().__init__(transforms)


class RealNVP(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 8, hidden_dim: int = 512):
        transforms: list[nn.Module] = []
        for layer in range(num_layers):
            transforms.append(AffineCoupling(dim, alternating_mask(dim, layer), hidden_dim=hidden_dim))
        super().__init__(transforms)


class GlowLite(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 6, hidden_dim: int = 512):
        transforms: list[nn.Module] = []
        for layer in range(num_layers):
            transforms.extend(
                [
                    ActNorm(dim),
                    InvertibleLinear(dim),
                    AffineCoupling(dim, alternating_mask(dim, layer), hidden_dim=hidden_dim),
                ]
            )
        super().__init__(transforms)


class MaskedAutoregressiveFlow(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 4, hidden_dim: int = 512):
        transforms = [MaskedAutoregressiveTransform(dim, hidden_dim=hidden_dim) for _ in range(num_layers)]
        super().__init__(transforms)


class InverseAutoregressiveFlow(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 4, hidden_dim: int = 512):
        transforms = [InverseAutoregressiveTransform(dim, hidden_dim=hidden_dim) for _ in range(num_layers)]
        super().__init__(transforms)


class NeuralSplineFlow(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 8, hidden_dim: int = 512):
        transforms: list[nn.Module] = []
        for layer in range(num_layers):
            transforms.append(SplineCoupling(dim, alternating_mask(dim, layer), hidden_dim=hidden_dim))
        super().__init__(transforms)


class PlanarFlow(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 16):
        transforms = [PlanarTransform(dim) for _ in range(num_layers)]
        super().__init__(transforms)


class RadialFlow(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 16):
        transforms = [RadialTransform(dim) for _ in range(num_layers)]
        super().__init__(transforms)


class FFJORDLite(NormalizingFlow):
    def __init__(self, dim: int = 784, hidden_dim: int = 256, ode_steps: int = 8):
        super().__init__([CNFTransform(dim, hidden_dim=hidden_dim, steps=ode_steps)])


class FlowPlusPlus(NormalizingFlow):
    def __init__(self, dim: int = 784, num_layers: int = 8, hidden_dim: int = 512):
        transforms: list[nn.Module] = []
        for layer in range(num_layers):
            transforms.extend(
                [
                    ActNorm(dim),
                    InvertibleLinear(dim),
                    SplineCoupling(dim, alternating_mask(dim, layer), hidden_dim=hidden_dim, num_terms=6),
                ]
            )
        super().__init__(transforms)
