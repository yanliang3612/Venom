from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.flows.factory import FLOW_VARIANTS, build_mnist_flow, checkpoint_config
from venom.utils import default_device, ensure_dir, seed_everything, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Venom normalizing flow models on MNIST.")
    parser.add_argument("--variant", choices=FLOW_VARIANTS, default="realnvp")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_flow"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def save_samples(model, epoch: int, out_dir: Path, device: torch.device) -> None:
    model.eval()
    samples = model.sample(64, device)
    save_image(unnormalize_to_zero_to_one(samples), out_dir / f"samples_{epoch:03d}.png", nrow=8)
    model.train()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir / args.variant)
    device = default_device()
    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)
    model, _ = build_mnist_flow(
        args.variant,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        ode_steps=args.ode_steps,
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for images, _ in progress:
            images = images.to(device)
            loss = model.training_loss(images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(bpd=f"{loss.item():.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": checkpoint_config(args.variant, args.hidden_dim, args.num_layers, args.ode_steps),
            },
            out_dir / f"model_{epoch:03d}.pt",
        )
        print(f"epoch={epoch} bits_per_dim={running_loss / len(loader):.4f}")

        if epoch % args.sample_every == 0:
            save_samples(model, epoch, out_dir, device)


if __name__ == "__main__":
    main()
