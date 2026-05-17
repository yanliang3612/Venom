from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from venom.diffusion.factory import build_mnist_diffusion
from venom.diffusion.guidance import classifier_guidance_gradient
from venom.diffusion.models import MNISTClassifier
from venom.diffusion.samplers import make_sampler
from venom.utils import default_device, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from a Venom diffusion, score, flow, or one-step checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("samples.png"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sampler", choices=["native", "ddpm", "ddim", "dpm-solver", "dpm-solver++", "edm"], default="native")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated labels, e.g. 0,1,2,3.")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--classifier-scale", type=float, default=1.0)
    return parser.parse_args()


def parse_labels(raw: str | None, num_samples: int, device: torch.device) -> torch.Tensor | None:
    if raw is None:
        return None
    labels = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not labels:
        return None
    repeated = (labels * ((num_samples + len(labels) - 1) // len(labels)))[:num_samples]
    return torch.tensor(repeated, device=device, dtype=torch.long)


def main() -> None:
    args = parse_args()
    device = default_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model, diffusion = build_mnist_diffusion(
        config["variant"],
        timesteps=config.get("timesteps", 1000),
        base_channels=config.get("base_channels", 64),
        class_dropout=config.get("class_dropout", 0.0),
        backbone=config.get("backbone", "unet"),
    )
    model.load_state_dict(checkpoint["model"])
    diffusion = diffusion.to(device).eval()
    labels = parse_labels(args.labels, args.num_samples, device)

    classifier_guidance_fn = None
    if args.classifier_checkpoint is not None:
        classifier_state = torch.load(args.classifier_checkpoint, map_location=device)
        classifier_config = classifier_state.get("config", {})
        classifier = MNISTClassifier(
            base_channels=classifier_config.get("base_channels", 64),
        ).to(device)
        classifier.load_state_dict(classifier_state["model"])
        classifier.eval()
        classifier_guidance_fn = classifier_guidance_gradient(classifier, scale=args.classifier_scale)

    continuous_variants = {
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
    }

    if config["variant"] in {"edm", "pfgm", "pfgm++"}:
        samples = diffusion.sample(args.num_samples, device, steps=args.sample_steps, y=labels)
    elif config["variant"] in {
        "rectified-flow",
        "flow-matching",
        "conditional-flow-matching",
        "ot-cfm",
        "stochastic-interpolants",
        "consistency",
        "shortcut",
        "meanflow",
    }:
        samples = diffusion.sample(args.num_samples, device, steps=args.sample_steps, y=labels)
    elif config["variant"] in {"score-sde-vp", "score-sde-ve", "score-sde-subvp"}:
        samples = diffusion.sample(args.num_samples, device, steps=args.sample_steps, corrector_steps=1, y=labels)
    elif config["variant"] in {"ncsn", "ncsnv2"}:
        steps_each = max(1, args.sample_steps // len(diffusion.sigmas))
        samples = diffusion.sample(args.num_samples, device, steps_each=steps_each, y=labels)
    elif args.sampler == "ddpm":
        samples = diffusion.sample(
            args.num_samples,
            device,
            y=labels,
            guidance_scale=args.guidance_scale,
            classifier_guidance_fn=classifier_guidance_fn,
        )
    else:
        sampler_name = "ddim" if args.sampler == "native" else args.sampler
        if config["variant"] in continuous_variants:
            raise ValueError("DDPM-family samplers can only be used with DDPM-family checkpoints")
        sampler = make_sampler(diffusion, sampler_name, steps=args.sample_steps, eta=args.eta)
        samples = sampler.sample(
            args.num_samples,
            device,
            y=labels,
            guidance_scale=args.guidance_scale,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(unnormalize_to_zero_to_one(samples), args.out, nrow=int(args.num_samples**0.5))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
