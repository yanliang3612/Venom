from __future__ import annotations

from dataclasses import dataclass

from venom.ebm.models import (
    ConditionalDeepEnergyModel,
    ConditionalRBM,
    ConvRBM,
    DeepEnergyModel,
    GaussianBernoulliRBM,
    JointEnergyModel,
    NCEEnergyModel,
    RBM,
    ScoreMatchingEBM,
    SlicedScoreMatchingEBM,
)


EBM_VARIANTS = (
    "rbm",
    "gaussian-rbm",
    "conditional-rbm",
    "conv-rbm",
    "deep-ebm",
    "conditional-ebm",
    "jem",
    "score-matching-ebm",
    "sliced-score-matching-ebm",
    "nce-ebm",
)


@dataclass(frozen=True)
class EBMConfig:
    variant: str
    family: str
    objective: str
    conditional: bool = False
    uses_replay_buffer: bool = False
    rbm_data_range: bool = False


def ebm_config(variant: str) -> EBMConfig:
    variant = variant.lower()
    if variant == "rbm":
        return EBMConfig(variant, family="rbm", objective="cd", rbm_data_range=True)
    if variant == "gaussian-rbm":
        return EBMConfig(variant, family="rbm", objective="cd", rbm_data_range=True)
    if variant == "conditional-rbm":
        return EBMConfig(variant, family="rbm", objective="cd", conditional=True, rbm_data_range=True)
    if variant == "conv-rbm":
        return EBMConfig(variant, family="rbm", objective="cd", rbm_data_range=True)
    if variant == "deep-ebm":
        return EBMConfig(variant, family="deep", objective="contrastive-divergence", uses_replay_buffer=True)
    if variant == "conditional-ebm":
        return EBMConfig(variant, family="deep", objective="contrastive-divergence", conditional=True, uses_replay_buffer=True)
    if variant == "jem":
        return EBMConfig(variant, family="jem", objective="joint-energy", conditional=True, uses_replay_buffer=True)
    if variant == "score-matching-ebm":
        return EBMConfig(variant, family="deep", objective="denoising-score-matching")
    if variant == "sliced-score-matching-ebm":
        return EBMConfig(variant, family="deep", objective="sliced-score-matching")
    if variant == "nce-ebm":
        return EBMConfig(variant, family="deep", objective="noise-contrastive-estimation")
    raise ValueError(f"variant must be one of: {', '.join(EBM_VARIANTS)}")


def build_mnist_ebm(variant: str, hidden_dim: int = 256, base_channels: int = 64):
    config = ebm_config(variant)
    if variant == "rbm":
        model = RBM(hidden_dim=hidden_dim)
    elif variant == "gaussian-rbm":
        model = GaussianBernoulliRBM(hidden_dim=hidden_dim)
    elif variant == "conditional-rbm":
        model = ConditionalRBM(hidden_dim=hidden_dim)
    elif variant == "conv-rbm":
        model = ConvRBM(hidden_channels=base_channels)
    elif variant == "deep-ebm":
        model = DeepEnergyModel(base_channels=base_channels)
    elif variant == "conditional-ebm":
        model = ConditionalDeepEnergyModel(base_channels=base_channels)
    elif variant == "jem":
        model = JointEnergyModel(base_channels=base_channels)
    elif variant == "score-matching-ebm":
        model = ScoreMatchingEBM(base_channels=base_channels)
    elif variant == "sliced-score-matching-ebm":
        model = SlicedScoreMatchingEBM(base_channels=base_channels)
    elif variant == "nce-ebm":
        model = NCEEnergyModel(base_channels=base_channels)
    else:
        raise ValueError(f"variant must be one of: {', '.join(EBM_VARIANTS)}")
    return model, config


def checkpoint_config(variant: str, hidden_dim: int, base_channels: int):
    config = ebm_config(variant)
    return {
        "family": "ebm",
        "variant": variant,
        "hidden_dim": hidden_dim,
        "base_channels": base_channels,
        "config": config.__dict__,
        "image_size": 28,
        "channels": 1,
    }
