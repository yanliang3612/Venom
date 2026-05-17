from __future__ import annotations

from dataclasses import dataclass

from venom.gan.models import DCGANDiscriminator, DCGANGenerator, MLPDiscriminator, MLPGenerator


GAN_VARIANTS = (
    "gan",
    "dcgan",
    "cgan",
    "acgan",
    "infogan",
    "lsgan",
    "wgan",
    "wgan-gp",
    "hinge-gan",
    "sn-gan",
)


@dataclass(frozen=True)
class GANConfig:
    variant: str
    loss: str = "vanilla"
    architecture: str = "dcgan"
    conditional: bool = False
    auxiliary_classifier: bool = False
    info_code_dim: int = 0
    use_gradient_penalty: bool = False
    use_weight_clipping: bool = False
    use_spectral_norm: bool = False
    critic_steps: int = 1


def gan_config(variant: str) -> GANConfig:
    variant = variant.lower()
    if variant == "gan":
        return GANConfig(variant=variant, architecture="mlp")
    if variant == "dcgan":
        return GANConfig(variant=variant, architecture="dcgan")
    if variant == "cgan":
        return GANConfig(variant=variant, architecture="dcgan", conditional=True)
    if variant == "acgan":
        return GANConfig(variant=variant, architecture="dcgan", conditional=True, auxiliary_classifier=True)
    if variant == "infogan":
        return GANConfig(variant=variant, architecture="dcgan", info_code_dim=2)
    if variant == "lsgan":
        return GANConfig(variant=variant, architecture="dcgan", loss="lsgan")
    if variant == "wgan":
        return GANConfig(variant=variant, architecture="dcgan", loss="wgan", use_weight_clipping=True, critic_steps=5)
    if variant == "wgan-gp":
        return GANConfig(variant=variant, architecture="dcgan", loss="wgan", use_gradient_penalty=True, critic_steps=5)
    if variant == "hinge-gan":
        return GANConfig(variant=variant, architecture="dcgan", loss="hinge")
    if variant == "sn-gan":
        return GANConfig(variant=variant, architecture="dcgan", loss="hinge", use_spectral_norm=True)
    raise ValueError(f"variant must be one of: {', '.join(GAN_VARIANTS)}")


def build_mnist_gan(
    variant: str,
    latent_dim: int = 128,
    base_channels: int = 64,
):
    config = gan_config(variant)
    num_classes = 10 if config.conditional else None
    code_dim = config.info_code_dim

    if config.architecture == "mlp":
        generator = MLPGenerator(latent_dim=latent_dim, num_classes=num_classes, code_dim=code_dim)
        discriminator = MLPDiscriminator(
            num_classes=num_classes,
            auxiliary_classifier=config.auxiliary_classifier,
            code_dim=code_dim,
        )
    else:
        generator = DCGANGenerator(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_classes=num_classes,
            code_dim=code_dim,
        )
        discriminator = DCGANDiscriminator(
            base_channels=base_channels,
            num_classes=num_classes,
            auxiliary_classifier=config.auxiliary_classifier,
            code_dim=code_dim,
            use_spectral_norm=config.use_spectral_norm,
        )
    return generator, discriminator, config


def checkpoint_config(variant: str, latent_dim: int, base_channels: int):
    config = gan_config(variant)
    return {
        "family": "gan",
        "variant": variant,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "config": config.__dict__,
        "image_size": 28,
        "channels": 1,
    }
