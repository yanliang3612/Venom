from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self, visible_dim: int = 784, hidden_dim: int = 256):
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(visible_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)

    def free_energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        v = self.flatten(x)
        visible_term = v @ self.visible_bias
        hidden_term = F.softplus(F.linear(v, self.weight, self.hidden_bias)).sum(dim=1)
        return -visible_term - hidden_term

    def sample_hidden(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(F.linear(self.flatten(x), self.weight, self.hidden_bias))
        return probs, torch.bernoulli(probs)

    def sample_visible(self, h: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(F.linear(h, self.weight.t(), self.visible_bias))
        return probs, torch.bernoulli(probs)

    @torch.no_grad()
    def gibbs(self, x: torch.Tensor, steps: int = 1, y: torch.Tensor | None = None) -> torch.Tensor:
        v = self.flatten(x)
        for _ in range(steps):
            _, h = self.sample_hidden(v, y)
            probs, v = self.sample_visible(h, y)
        return probs.reshape(x.shape[0], 1, 28, 28)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, steps: int = 100, y: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.rand(batch_size, 1, 28, 28, device=device)
        return self.gibbs(x, steps=steps, y=y)


class GaussianBernoulliRBM(RBM):
    def __init__(self, visible_dim: int = 784, hidden_dim: int = 256, sigma: float = 1.0):
        super().__init__(visible_dim=visible_dim, hidden_dim=hidden_dim)
        self.sigma = sigma

    def free_energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        v = self.flatten(x)
        centered = (v - self.visible_bias) / self.sigma
        visible_term = 0.5 * centered.pow(2).sum(dim=1)
        hidden_term = F.softplus(F.linear(v / self.sigma, self.weight, self.hidden_bias)).sum(dim=1)
        return visible_term - hidden_term

    def sample_visible(self, h: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mean = F.linear(h, self.weight.t(), self.visible_bias)
        sample = mean + self.sigma * torch.randn_like(mean)
        return mean.clamp(0.0, 1.0), sample.clamp(0.0, 1.0)


class ConditionalRBM(RBM):
    def __init__(self, visible_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__(visible_dim=visible_dim, hidden_dim=hidden_dim)
        self.num_classes = num_classes
        self.label_to_hidden = nn.Embedding(num_classes, hidden_dim)
        self.label_to_visible = nn.Embedding(num_classes, visible_dim)

    def _biases(self, y: torch.Tensor | None, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if y is None:
            y = torch.randint(0, self.num_classes, (batch_size,), device=device)
        return self.label_to_visible(y.long()), self.label_to_hidden(y.long())

    def free_energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        v = self.flatten(x)
        label_visible, label_hidden = self._biases(y, v.shape[0], v.device)
        visible_bias = self.visible_bias.unsqueeze(0) + label_visible
        hidden_bias = self.hidden_bias.unsqueeze(0) + label_hidden
        visible_term = (v * visible_bias).sum(dim=1)
        hidden_term = F.softplus(F.linear(v, self.weight, None) + hidden_bias).sum(dim=1)
        return -visible_term - hidden_term

    def sample_hidden(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.flatten(x)
        _, hidden_bias = self._biases(y, v.shape[0], v.device)
        probs = torch.sigmoid(F.linear(v, self.weight, None) + self.hidden_bias + hidden_bias)
        return probs, torch.bernoulli(probs)

    def sample_visible(self, h: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        visible_bias, _ = self._biases(y, h.shape[0], h.device)
        logits = F.linear(h, self.weight.t(), None) + self.visible_bias + visible_bias
        probs = torch.sigmoid(logits)
        return probs, torch.bernoulli(probs)


class ConvRBM(nn.Module):
    def __init__(self, channels: int = 1, hidden_channels: int = 64, kernel_size: int = 5):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(hidden_channels, channels, kernel_size, kernel_size) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(channels))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_channels))

    def free_energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        hidden_logits = F.conv2d(x, self.weight, self.hidden_bias)
        visible_term = (x * self.visible_bias.view(1, -1, 1, 1)).flatten(1).sum(dim=1)
        hidden_term = F.softplus(hidden_logits).flatten(1).sum(dim=1)
        return -visible_term - hidden_term

    def sample_hidden(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(F.conv2d(x, self.weight, self.hidden_bias))
        return probs, torch.bernoulli(probs)

    def sample_visible(self, h: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.conv_transpose2d(h, self.weight, padding=0)
        logits = logits + self.visible_bias.view(1, -1, 1, 1)
        probs = torch.sigmoid(logits)
        return probs, torch.bernoulli(probs)

    @torch.no_grad()
    def gibbs(self, x: torch.Tensor, steps: int = 1, y: torch.Tensor | None = None) -> torch.Tensor:
        v = x
        for _ in range(steps):
            _, h = self.sample_hidden(v, y)
            probs, v = self.sample_visible(h, y)
        return probs

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, steps: int = 100, y: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.rand(batch_size, 1, 28, 28, device=device)
        return self.gibbs(x, steps=steps, y=y)


class DeepEnergyModel(nn.Module):
    def __init__(self, channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Linear(base_channels * 4 * 7 * 7, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.energy(x, y)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, steps: int = 100, y: torch.Tensor | None = None) -> torch.Tensor:
        from venom.ebm.samplers import sgld_sample

        x = torch.empty(batch_size, 1, 28, 28, device=device).uniform_(-1.0, 1.0)
        return sgld_sample(self, x, steps=steps, y=y)


class ConditionalDeepEnergyModel(DeepEnergyModel):
    def __init__(self, channels: int = 1, base_channels: int = 64, num_classes: int = 10):
        super().__init__(channels=channels, base_channels=base_channels)
        self.num_classes = num_classes
        feature_dim = base_channels * 4 * 7 * 7
        self.label_energy = nn.Embedding(num_classes, feature_dim)
        self.label_bias = nn.Embedding(num_classes, 1)

    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encode(x)
        base = self.head(h).squeeze(1)
        if y is None:
            return base
        label_term = (h * self.label_energy(y.long())).sum(dim=1)
        label_bias = self.label_bias(y.long()).squeeze(1)
        return base - label_term - label_bias


class JointEnergyModel(nn.Module):
    def __init__(self, channels: int = 1, base_channels: int = 64, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(base_channels * 4 * 7 * 7, num_classes)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.logits(x)
        if y is None:
            return -torch.logsumexp(logits, dim=1)
        return -logits.gather(1, y.long().view(-1, 1)).squeeze(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.energy(x, y)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, steps: int = 100, y: torch.Tensor | None = None) -> torch.Tensor:
        from venom.ebm.samplers import sgld_sample

        x = torch.empty(batch_size, 1, 28, 28, device=device).uniform_(-1.0, 1.0)
        return sgld_sample(self, x, steps=steps, y=y)


class ScoreMatchingEBM(DeepEnergyModel):
    pass


class SlicedScoreMatchingEBM(DeepEnergyModel):
    pass


class NCEEnergyModel(DeepEnergyModel):
    pass
