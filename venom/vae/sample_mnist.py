from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from venom.utils import default_device, unnormalize_to_zero_to_one
from venom.vae.factory import build_mnist_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from a Venom VAE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("vae_samples.png"))
    parser.add_argument("--num-samples", type=int, default=64)
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
    config = checkpoint["config"]
    model = build_mnist_vae(
        config["variant"],
        latent_dim=config.get("latent_dim", 64),
        base_channels=config.get("base_channels", 32),
        beta=config.get("beta", 4.0),
        importance_samples=config.get("importance_samples", 5),
        codebook_size=config.get("codebook_size", 512),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    labels = parse_labels(args.labels, args.num_samples, device)

    samples = model.sample(args.num_samples, device, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(unnormalize_to_zero_to_one(samples), args.out, nrow=int(args.num_samples**0.5))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
