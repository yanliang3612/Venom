from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from venom.flows.factory import build_mnist_flow
from venom.utils import default_device, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from a Venom normalizing flow checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("flow_samples.png"))
    parser.add_argument("--num-samples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = default_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model, _ = build_mnist_flow(
        config["variant"],
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 8),
        ode_steps=config.get("ode_steps", 8),
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    samples = model.sample(args.num_samples, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(unnormalize_to_zero_to_one(samples), args.out, nrow=int(args.num_samples**0.5))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
