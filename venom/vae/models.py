from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).flatten(1).sum(dim=1)


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2 * math.pi) + logvar + (x - mu).pow(2) / logvar.exp()).flatten(1).sum(dim=1)


class VAE(nn.Module):
    """Fully connected image VAE for small images such as MNIST."""

    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        beta: float = 1.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_dim = image_size * image_size * channels

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-20, 10)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = self.decoder(z)
        return x.reshape(z.shape[0], self.channels, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        recon, mu, logvar = self.forward(x, y)
        recon_loss = F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1)
        return (recon_loss + self.beta * kl_normal(mu, logvar)).mean()

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, y: torch.Tensor | None = None) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z, y).clamp(-1, 1)


class ConvVAE(VAE):
    """Convolutional VAE for image generation."""

    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        latent_dim: int = 64,
        base_channels: int = 32,
        beta: float = 1.0,
    ):
        nn.Module.__init__(self)
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.beta = beta
        self.base_channels = base_channels
        self.feature_size = image_size // 4
        feature_dim = base_channels * 4 * self.feature_size * self.feature_size

        self.encoder_net = nn.Sequential(
            nn.Conv2d(channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, feature_dim)
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        return self.mu(h), self.logvar(h).clamp(-20, 10)

    def decode(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.reshape(z.shape[0], self.base_channels * 4, self.feature_size, self.feature_size)
        return self.decoder_net(h)


class BetaVAE(ConvVAE):
    def __init__(self, *args, beta: float = 4.0, **kwargs):
        super().__init__(*args, beta=beta, **kwargs)


class CVAE(ConvVAE):
    """Class-conditional convolutional VAE."""

    def __init__(self, *args, num_classes: int = 10, label_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, label_dim)
        feature_dim = self.base_channels * 4 * self.feature_size * self.feature_size
        self.mu = nn.Linear(feature_dim + label_dim, self.latent_dim)
        self.logvar = nn.Linear(feature_dim + label_dim, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim + label_dim, feature_dim)

    def _labels(self, y: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if y is None:
            y = torch.randint(0, self.num_classes, (batch_size,), device=device)
        return self.label_emb(y.long())

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        label = self._labels(y, x.shape[0], x.device)
        h = torch.cat([h, label], dim=1)
        return self.mu(h), self.logvar(h).clamp(-20, 10)

    def decode(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        label = self._labels(y, z.shape[0], z.device)
        h = self.decoder_input(torch.cat([z, label], dim=1))
        h = h.reshape(z.shape[0], self.base_channels * 4, self.feature_size, self.feature_size)
        return self.decoder_net(h)


class IWAE(ConvVAE):
    """Importance Weighted Autoencoder objective using a ConvVAE backbone."""

    def __init__(self, *args, importance_samples: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_samples = importance_samples

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        mu, logvar = self.encode(x, y)
        log_weights = []
        for _ in range(self.importance_samples):
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, y)
            log_px_z = -0.5 * F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1)
            log_pz = gaussian_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))
            log_qz_x = gaussian_log_prob(z, mu, logvar)
            log_weights.append(log_px_z + log_pz - log_qz_x)
        log_weights = torch.stack(log_weights, dim=0)
        return -(torch.logsumexp(log_weights, dim=0) - math.log(self.importance_samples)).mean()


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.reshape(-1, self.embedding_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).reshape_as(z_perm).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = z + (quantized - z).detach()
        return quantized, loss, indices.reshape(z.shape[0], z.shape[2], z.shape[3])


class VQVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        embedding_dim: int = 64,
        codebook_size: int = 512,
        base_channels: int = 32,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.feature_size = image_size // 4
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, embedding_dim, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
        )
        self.quantizer = VectorQuantizer(codebook_size, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        return self.decoder(z_q), vq_loss, indices

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        recon, vq_loss, _ = self.forward(x, y)
        recon_loss = F.mse_loss(recon, x)
        return recon_loss + vq_loss

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, y: torch.Tensor | None = None) -> torch.Tensor:
        indices = torch.randint(
            0,
            self.quantizer.codebook_size,
            (batch_size, self.feature_size, self.feature_size),
            device=device,
        )
        z = self.quantizer.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z).clamp(-1, 1)


class HierarchicalVAE(ConvVAE):
    """Two-level hierarchical VAE, useful as a compact Ladder-VAE-style baseline."""

    def __init__(self, *args, top_latent_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_latent_dim = top_latent_dim
        feature_dim = self.base_channels * 4 * self.feature_size * self.feature_size
        self.top_mu = nn.Linear(feature_dim, top_latent_dim)
        self.top_logvar = nn.Linear(feature_dim, top_latent_dim)
        self.bottom_mu = nn.Linear(feature_dim + top_latent_dim, self.latent_dim)
        self.bottom_logvar = nn.Linear(feature_dim + top_latent_dim, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim + top_latent_dim, feature_dim)

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None):
        h = self.encoder_net(x)
        mu2 = self.top_mu(h)
        logvar2 = self.top_logvar(h).clamp(-20, 10)
        z2 = self.reparameterize(mu2, logvar2)
        h_bottom = torch.cat([h, z2], dim=1)
        mu1 = self.bottom_mu(h_bottom)
        logvar1 = self.bottom_logvar(h_bottom).clamp(-20, 10)
        return mu1, logvar1, mu2, logvar2

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        mu1, logvar1, mu2, logvar2 = self.encode(x, y)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        recon = self.decode(torch.cat([z1, z2], dim=1), y)
        return recon, mu1, logvar1, mu2, logvar2

    def decode(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.reshape(z.shape[0], self.base_channels * 4, self.feature_size, self.feature_size)
        return self.decoder_net(h)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        recon, mu1, logvar1, mu2, logvar2 = self.forward(x, y)
        recon_loss = F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1)
        kl = kl_normal(mu1, logvar1) + kl_normal(mu2, logvar2)
        return (recon_loss + self.beta * kl).mean()

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, y: torch.Tensor | None = None) -> torch.Tensor:
        z1 = torch.randn(batch_size, self.latent_dim, device=device)
        z2 = torch.randn(batch_size, self.top_latent_dim, device=device)
        return self.decode(torch.cat([z1, z2], dim=1), y).clamp(-1, 1)


LadderVAE = HierarchicalVAE


class PlanarFlow(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        linear = z @ self.w + self.b
        h = torch.tanh(linear)
        z_next = z + self.u * h[:, None]
        psi = (1 - h.pow(2))[:, None] * self.w
        log_abs_det = torch.log(torch.abs(1 + psi @ self.u[:, None]).squeeze(1).clamp(min=1e-6))
        return z_next, log_abs_det


class FlowVAE(ConvVAE):
    """ConvVAE with planar normalizing flows in the variational posterior."""

    def __init__(self, *args, num_flows: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.flows = nn.ModuleList([PlanarFlow(self.latent_dim) for _ in range(num_flows)])

    def flow(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, delta = flow(z)
            log_det = log_det + delta
        return z, log_det

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        mu, logvar = self.encode(x, y)
        z0 = self.reparameterize(mu, logvar)
        zk, log_det = self.flow(z0)
        recon = self.decode(zk, y)
        return recon, mu, logvar, z0, zk, log_det

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        recon, mu, logvar, z0, zk, log_det = self.forward(x, y)
        recon_loss = F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1)
        log_q0 = gaussian_log_prob(z0, mu, logvar)
        log_qk = log_q0 - log_det
        log_pzk = gaussian_log_prob(zk, torch.zeros_like(zk), torch.zeros_like(zk))
        return (recon_loss + log_qk - log_pzk).mean()
