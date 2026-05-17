from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from venom.ebm.factory import build_mnist_ebm
from venom.ebm.samplers import sgld_sample
from venom.ebm.train_mnist import from_rbm_range
from venom.utils import default_device, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from a Venom EBM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ebm_samples.png"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--labels", type=str, default=None)
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
    ckpt_config = checkpoint["config"]
    model, config = build_mnist_ebm(
        ckpt_config["variant"],
        hidden_dim=ckpt_config.get("hidden_dim", 256),
        base_channels=ckpt_config.get("base_channels", 64),
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    labels = parse_labels(args.labels, args.num_samples, device)
    if config.conditional and labels is None:
        labels = torch.arange(args.num_samples, device=device) % 10

    if config.family == "rbm":
        samples = model.sample(args.num_samples, device, steps=args.steps, y=labels)
        samples = from_rbm_range(samples)
    else:
        init = torch.empty(args.num_samples, 1, 28, 28, device=device).uniform_(-1.0, 1.0)
        samples = sgld_sample(
            model,
            init,
            steps=args.steps,
            step_size=args.step_size,
            noise_scale=args.noise_scale,
            y=labels,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(unnormalize_to_zero_to_one(samples), args.out, nrow=int(args.num_samples**0.5))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
