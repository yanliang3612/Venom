from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.utils import default_device, ensure_dir, seed_everything, unnormalize_to_zero_to_one
from venom.vae.factory import VAE_VARIANTS, build_mnist_vae, checkpoint_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Venom VAE models on MNIST.")
    parser.add_argument("--variant", choices=VAE_VARIANTS, default="conv-vae")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_vae"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--importance-samples", type=int, default=5)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def save_samples(model, variant: str, epoch: int, out_dir: Path, device: torch.device) -> None:
    model.eval()
    labels = torch.arange(64, device=device) % 10 if variant == "cvae" else None
    samples = model.sample(64, device, labels)
    save_image(unnormalize_to_zero_to_one(samples), out_dir / f"samples_{epoch:03d}.png", nrow=8)
    model.train()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir / args.variant)
    device = default_device()
    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)
    model = build_mnist_vae(
        args.variant,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        beta=args.beta,
        importance_samples=args.importance_samples,
        codebook_size=args.codebook_size,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            use_labels = labels if args.variant == "cvae" else None
            loss = model.training_loss(images, use_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        config = checkpoint_config(
            args.variant,
            args.latent_dim,
            args.base_channels,
            args.beta,
            args.importance_samples,
            args.codebook_size,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            },
            out_dir / f"model_{epoch:03d}.pt",
        )
        print(f"epoch={epoch} loss={running_loss / len(loader):.4f}")

        if epoch % args.sample_every == 0:
            save_samples(model, args.variant, epoch, out_dir, device)


if __name__ == "__main__":
    main()
