from __future__ import annotations

import torch
import torch.nn.functional as F


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    return images.flatten(1)


def logit_preprocess(
    images: torch.Tensor,
    alpha: float = 1e-5,
    dequantize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (images + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    if dequantize:
        x = (x * 255.0 + torch.rand_like(x)) / 256.0
    x = alpha + (1.0 - 2.0 * alpha) * x
    logits = torch.log(x) - torch.log1p(-x)
    logdet = torch.log(torch.tensor(1.0 - 2.0 * alpha, device=x.device, dtype=x.dtype))
    logdet = logdet - F.logsigmoid(logits) - F.logsigmoid(-logits)
    return flatten_images(logits), flatten_images(logdet).sum(dim=1)


def inverse_logit_preprocess(flat: torch.Tensor, image_size: int = 28, channels: int = 1, alpha: float = 1e-5) -> torch.Tensor:
    x = torch.sigmoid(flat)
    x = (x - alpha) / (1.0 - 2.0 * alpha)
    x = x.clamp(0.0, 1.0)
    return x.reshape(flat.shape[0], channels, image_size, image_size) * 2.0 - 1.0


def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (z.pow(2) + torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))).sum(dim=1)


def bits_per_dim(negative_log_likelihood: torch.Tensor, dim: int) -> torch.Tensor:
    return negative_log_likelihood / (dim * torch.log(torch.tensor(2.0, device=negative_log_likelihood.device)))
