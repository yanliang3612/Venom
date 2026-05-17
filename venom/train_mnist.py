from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.factory import VARIANTS, build_mnist_diffusion, checkpoint_config
from venom.samplers import make_sampler
from venom.utils import default_device, ensure_dir, seed_everything, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train venom models on MNIST.")
    parser.add_argument("--variant", choices=VARIANTS, default="ddpm")
    parser.add_argument("--backbone", choices=["unet", "dit"], default="unet")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def save_samples(diffusion, variant: str, epoch: int, out_dir: Path, device: torch.device, sample_steps: int) -> None:
    diffusion.eval()
    if variant in {"edm", "pfgm", "pfgm++"}:
        samples = diffusion.sample(batch_size=64, device=device, steps=sample_steps)
    elif variant in {
        "rectified-flow",
        "flow-matching",
        "conditional-flow-matching",
        "ot-cfm",
        "stochastic-interpolants",
        "consistency",
        "shortcut",
        "meanflow",
    }:
        samples = diffusion.sample(batch_size=64, device=device, steps=sample_steps)
    elif variant in {"score-sde-vp", "score-sde-ve", "score-sde-subvp"}:
        samples = diffusion.sample(batch_size=64, device=device, steps=sample_steps, corrector_steps=1)
    elif variant in {"ncsn", "ncsnv2"}:
        steps_each = max(1, sample_steps // len(diffusion.sigmas))
        samples = diffusion.sample(batch_size=64, device=device, steps_each=steps_each)
    else:
        labels = None
        guidance_scale = 1.0
        if variant in {"adm", "cfg"}:
            labels = torch.arange(64, device=device) % 10
            guidance_scale = 3.0 if variant == "cfg" else 1.0
        sampler = make_sampler(diffusion, "ddim", steps=sample_steps)
        samples = sampler.sample(batch_size=64, device=device, y=labels, guidance_scale=guidance_scale)
    save_image(unnormalize_to_zero_to_one(samples), out_dir / f"samples_{epoch:03d}.png", nrow=8)
    diffusion.train()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir / args.variant)

    device = default_device()
    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)
    model, diffusion = build_mnist_diffusion(
        args.variant,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        class_dropout=args.class_dropout,
        backbone=args.backbone,
    )
    diffusion = diffusion.to(device)
    optimizer = AdamW(diffusion.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        running_loss = 0.0

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            use_labels = labels if args.variant in {"adm", "cfg"} else None
            loss = diffusion.training_loss(images, use_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": checkpoint_config(
                args.variant,
                args.timesteps,
                args.base_channels,
                args.class_dropout,
                args.backbone,
            ),
        }
        torch.save(checkpoint, out_dir / f"model_{epoch:03d}.pt")
        print(f"epoch={epoch} loss={avg_loss:.4f}")

        if epoch % args.sample_every == 0:
            save_samples(diffusion, args.variant, epoch, out_dir, device, args.sample_steps)


if __name__ == "__main__":
    main()
