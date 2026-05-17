from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.diffusion import GaussianDiffusion
from venom.models import MNISTClassifier, UNet2D
from venom.utils import default_device, ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a noised MNIST classifier for ADM classifier guidance.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist/classifier"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = default_device()

    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)
    classifier = MNISTClassifier(base_channels=args.base_channels).to(device)

    # A dummy diffusion object provides the same q(x_t | x_0) corruption as the generator.
    dummy_model = UNet2D(base_channels=8)
    diffusion = GaussianDiffusion(dummy_model, timesteps=args.timesteps, beta_schedule="cosine").to(device)
    optimizer = AdamW(classifier.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"classifier epoch {epoch}/{args.epochs}")
        correct = 0
        seen = 0
        running_loss = 0.0

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            timesteps = torch.randint(0, args.timesteps, (images.shape[0],), device=device)
            noised = diffusion.q_sample(images, timesteps)
            logits = classifier(noised, timesteps)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.numel()
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / seen:.3f}")

        checkpoint = {
            "model": classifier.state_dict(),
            "epoch": epoch,
            "config": {
                "timesteps": args.timesteps,
                "base_channels": args.base_channels,
            },
        }
        torch.save(checkpoint, out_dir / f"classifier_{epoch:03d}.pt")
        print(f"epoch={epoch} loss={running_loss / len(loader):.4f} acc={correct / seen:.4f}")


if __name__ == "__main__":
    main()
