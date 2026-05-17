from __future__ import annotations

from venom.diffusion import GaussianDiffusion
from venom.diffusion.edm import EDMDiffusion
from venom.diffusion.flow_matching import FlowMatchingDiffusion
from venom.diffusion.models import DiT, UNet2D
from venom.diffusion.ncsn import NCSNDiffusion
from venom.diffusion.one_step import ConsistencyModel, MeanFlow, ShortcutModel
from venom.diffusion.pfgm import PFGMDiffusion, PFGMPlusPlusDiffusion
from venom.diffusion.score_sde import ScoreSDEDiffusion, SubVPSDE, VESDE, VPSDE


VARIANTS = (
    "ddpm",
    "improved-ddpm",
    "adm",
    "cfg",
    "edm",
    "ncsn",
    "ncsnv2",
    "score-sde-vp",
    "score-sde-ve",
    "score-sde-subvp",
    "pfgm",
    "pfgm++",
    "rectified-flow",
    "flow-matching",
    "conditional-flow-matching",
    "ot-cfm",
    "stochastic-interpolants",
    "consistency",
    "shortcut",
    "meanflow",
)


def _build_backbone(
    backbone: str,
    image_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64,
    num_classes: int | None = None,
    class_dropout: float = 0.0,
):
    if backbone == "unet":
        return UNet2D(
            image_channels=image_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            class_dropout=class_dropout,
        )
    if backbone == "dit":
        hidden_size = ((max(96, base_channels * 3) + 5) // 6) * 6
        num_heads = 6
        return DiT(
            image_channels=image_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_classes=num_classes,
            class_dropout=class_dropout,
        )
    raise ValueError("backbone must be one of: unet, dit")


def build_mnist_diffusion(
    variant: str,
    timesteps: int = 1000,
    base_channels: int = 64,
    class_dropout: float = 0.0,
    backbone: str = "unet",
):
    variant = variant.lower()

    if variant == "ddpm":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = GaussianDiffusion(model, timesteps=timesteps, beta_schedule="linear")
    elif variant == "improved-ddpm":
        model = _build_backbone(backbone, out_channels=2, base_channels=base_channels)
        diffusion = GaussianDiffusion(
            model,
            timesteps=timesteps,
            beta_schedule="cosine",
            variance_type="learned_range",
        )
    elif variant == "adm":
        model = _build_backbone(backbone, out_channels=2, base_channels=base_channels, num_classes=10)
        diffusion = GaussianDiffusion(
            model,
            timesteps=timesteps,
            beta_schedule="cosine",
            variance_type="learned_range",
        )
    elif variant == "cfg":
        model = _build_backbone(
            backbone,
            out_channels=2,
            base_channels=base_channels,
            num_classes=10,
            class_dropout=class_dropout,
        )
        diffusion = GaussianDiffusion(
            model,
            timesteps=timesteps,
            beta_schedule="cosine",
            variance_type="learned_range",
        )
    elif variant == "edm":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = EDMDiffusion(model)
    elif variant == "ncsn":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = NCSNDiffusion(model, num_sigmas=10, version="ncsn")
    elif variant == "ncsnv2":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = NCSNDiffusion(model, num_sigmas=30, version="ncsnv2")
    elif variant == "score-sde-vp":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = ScoreSDEDiffusion(model, VPSDE())
    elif variant == "score-sde-ve":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = ScoreSDEDiffusion(model, VESDE())
    elif variant == "score-sde-subvp":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = ScoreSDEDiffusion(model, SubVPSDE())
    elif variant == "pfgm":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = PFGMDiffusion(model)
    elif variant == "pfgm++":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = PFGMPlusPlusDiffusion(model)
    elif variant in {
        "rectified-flow",
        "flow-matching",
        "conditional-flow-matching",
        "ot-cfm",
        "stochastic-interpolants",
    }:
        model = _build_backbone(backbone, base_channels=base_channels)
        path_sigma = 0.05 if variant in {"flow-matching", "conditional-flow-matching"} else 0.0
        diffusion = FlowMatchingDiffusion(model, variant=variant, path_sigma=path_sigma)
    elif variant == "consistency":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = ConsistencyModel(model)
    elif variant == "shortcut":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = ShortcutModel(model)
    elif variant == "meanflow":
        model = _build_backbone(backbone, base_channels=base_channels)
        diffusion = MeanFlow(model)
    else:
        raise ValueError(f"variant must be one of: {', '.join(VARIANTS)}")

    return model, diffusion


def checkpoint_config(
    variant: str,
    timesteps: int,
    base_channels: int,
    class_dropout: float,
    backbone: str = "unet",
):
    return {
        "variant": variant,
        "timesteps": timesteps,
        "base_channels": base_channels,
        "class_dropout": class_dropout,
        "backbone": backbone,
        "image_size": 28,
        "channels": 1,
    }
