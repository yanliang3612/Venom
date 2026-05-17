from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from venom.gan.factory import build_mnist_gan
from venom.gan.train_mnist import sample_latents
from venom.utils import default_device, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from a Venom GAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("gan_samples.png"))
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
    ckpt_config = checkpoint["config"]
    generator, _, config = build_mnist_gan(
        ckpt_config["variant"],
        latent_dim=ckpt_config.get("latent_dim", 128),
        base_channels=ckpt_config.get("base_channels", 64),
    )
    generator = generator.to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    labels = parse_labels(args.labels, args.num_samples, device)
    if config.conditional and labels is None:
        labels = torch.arange(args.num_samples, device=device) % 10
    z, code = sample_latents(args.num_samples, ckpt_config.get("latent_dim", 128), config.info_code_dim, device)

    with torch.no_grad():
        samples = generator(z, labels, code)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(unnormalize_to_zero_to_one(samples), args.out, nrow=int(args.num_samples**0.5))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
