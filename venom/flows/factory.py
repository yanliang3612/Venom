from __future__ import annotations

from dataclasses import dataclass

from venom.flows.models import (
    FFJORDLite,
    FlowPlusPlus,
    GlowLite,
    InverseAutoregressiveFlow,
    MaskedAutoregressiveFlow,
    NICE,
    NeuralSplineFlow,
    PlanarFlow,
    RadialFlow,
    RealNVP,
)


FLOW_VARIANTS = (
    "planar-flow",
    "radial-flow",
    "nice",
    "realnvp",
    "glow",
    "maf",
    "iaf",
    "neural-spline-flow",
    "ffjord",
    "flow++",
)


@dataclass(frozen=True)
class FlowConfig:
    variant: str
    family: str = "normalizing-flow"
    exact_inverse: bool = True
    exact_logdet: bool = True
    notes: str = ""


def flow_config(variant: str) -> FlowConfig:
    variant = variant.lower()
    if variant == "planar-flow":
        return FlowConfig(variant, exact_inverse=False, notes="Planar inverse uses fixed-point iterations.")
    if variant == "radial-flow":
        return FlowConfig(variant, exact_inverse=False, notes="Radial inverse uses fixed-point iterations.")
    if variant == "nice":
        return FlowConfig(variant, notes="Additive coupling flow.")
    if variant == "realnvp":
        return FlowConfig(variant, notes="Affine coupling flow.")
    if variant == "glow":
        return FlowConfig(variant, notes="Glow-style ActNorm + invertible linear + affine coupling.")
    if variant == "maf":
        return FlowConfig(variant, notes="Masked autoregressive flow.")
    if variant == "iaf":
        return FlowConfig(variant, notes="Inverse-autoregressive educational flow.")
    if variant == "neural-spline-flow":
        return FlowConfig(variant, notes="Monotone spline-style coupling flow.")
    if variant == "ffjord":
        return FlowConfig(variant, exact_logdet=False, notes="Continuous normalizing flow with Hutchinson trace estimates.")
    if variant == "flow++":
        return FlowConfig(variant, notes="Flow++-style ActNorm + invertible linear + nonlinear coupling.")
    raise ValueError(f"variant must be one of: {', '.join(FLOW_VARIANTS)}")


def build_mnist_flow(
    variant: str,
    hidden_dim: int = 512,
    num_layers: int = 8,
    ode_steps: int = 8,
):
    variant = variant.lower()
    dim = 28 * 28
    config = flow_config(variant)
    if variant == "planar-flow":
        model = PlanarFlow(dim=dim, num_layers=num_layers)
    elif variant == "radial-flow":
        model = RadialFlow(dim=dim, num_layers=num_layers)
    elif variant == "nice":
        model = NICE(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    elif variant == "realnvp":
        model = RealNVP(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    elif variant == "glow":
        model = GlowLite(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    elif variant == "maf":
        model = MaskedAutoregressiveFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    elif variant == "iaf":
        model = InverseAutoregressiveFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    elif variant == "neural-spline-flow":
        model = NeuralSplineFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    elif variant == "ffjord":
        model = FFJORDLite(dim=dim, hidden_dim=hidden_dim, ode_steps=ode_steps)
    elif variant == "flow++":
        model = FlowPlusPlus(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"variant must be one of: {', '.join(FLOW_VARIANTS)}")
    return model, config


def checkpoint_config(variant: str, hidden_dim: int, num_layers: int, ode_steps: int):
    config = flow_config(variant)
    return {
        "family": "flow",
        "variant": variant,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "ode_steps": ode_steps,
        "config": config.__dict__,
        "image_size": 28,
        "channels": 1,
    }
