from __future__ import annotations

from venom.vae.models import BetaVAE, CVAE, ConvVAE, FlowVAE, HierarchicalVAE, IWAE, VAE, VQVAE


VAE_VARIANTS = (
    "vae",
    "conv-vae",
    "beta-vae",
    "cvae",
    "iwae",
    "vq-vae",
    "ladder-vae",
    "hierarchical-vae",
    "flow-vae",
)


def build_mnist_vae(
    variant: str,
    latent_dim: int = 64,
    base_channels: int = 32,
    beta: float = 4.0,
    importance_samples: int = 5,
    codebook_size: int = 512,
):
    variant = variant.lower()

    if variant == "vae":
        return VAE(latent_dim=latent_dim)
    if variant == "conv-vae":
        return ConvVAE(latent_dim=latent_dim, base_channels=base_channels)
    if variant == "beta-vae":
        return BetaVAE(latent_dim=latent_dim, base_channels=base_channels, beta=beta)
    if variant == "cvae":
        return CVAE(latent_dim=latent_dim, base_channels=base_channels)
    if variant == "iwae":
        return IWAE(latent_dim=latent_dim, base_channels=base_channels, importance_samples=importance_samples)
    if variant == "vq-vae":
        return VQVAE(embedding_dim=latent_dim, base_channels=base_channels, codebook_size=codebook_size)
    if variant in {"ladder-vae", "hierarchical-vae"}:
        return HierarchicalVAE(latent_dim=latent_dim, base_channels=base_channels)
    if variant == "flow-vae":
        return FlowVAE(latent_dim=latent_dim, base_channels=base_channels)

    raise ValueError(f"variant must be one of: {', '.join(VAE_VARIANTS)}")


def checkpoint_config(
    variant: str,
    latent_dim: int,
    base_channels: int,
    beta: float,
    importance_samples: int,
    codebook_size: int,
):
    return {
        "family": "vae",
        "variant": variant,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "beta": beta,
        "importance_samples": importance_samples,
        "codebook_size": codebook_size,
        "image_size": 28,
        "channels": 1,
    }
