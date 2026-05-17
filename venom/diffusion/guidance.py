from __future__ import annotations

import torch
import torch.nn.functional as F


def classifier_guidance_gradient(classifier, scale: float = 1.0):
    """Return a gradient function for ADM-style classifier guidance.

    The returned function computes scale * grad_x log p(y | x_t, t). DDPM
    sampling then adds variance * gradient to the reverse-process mean.
    """

    def grad_fn(x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, timesteps)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(1, y[:, None]).sum()
            grad = torch.autograd.grad(selected, x_in)[0]
        return grad * scale

    return grad_fn
