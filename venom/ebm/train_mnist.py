from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.ebm.factory import EBM_VARIANTS, build_mnist_ebm, checkpoint_config
from venom.ebm.losses import (
    cd_loss,
    contrastive_energy_loss,
    denoising_score_matching_loss,
    nce_loss,
    pcd_loss,
    sliced_score_matching_loss,
)
from venom.ebm.samplers import sgld_sample
from venom.utils import default_device, ensure_dir, seed_everything, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Venom energy-based models on MNIST.")
    parser.add_argument("--variant", choices=EBM_VARIANTS, default="deep-ebm")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_ebm"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--cd-steps", type=int, default=1)
    parser.add_argument("--sgld-steps", type=int, default=40)
    parser.add_argument("--sgld-step-size", type=float, default=1.0)
    parser.add_argument("--sgld-noise-scale", type=float, default=0.01)
    parser.add_argument("--score-sigma", type=float, default=0.1)
    parser.add_argument("--classifier-weight", type=float, default=1.0)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_rbm_range(images: torch.Tensor) -> torch.Tensor:
    return (images + 1.0) * 0.5


def from_rbm_range(images: torch.Tensor) -> torch.Tensor:
    return images * 2.0 - 1.0


def sample_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size, device=device) % 10


@torch.no_grad()
def save_samples(model, config, epoch: int, out_dir: Path, device: torch.device, args: argparse.Namespace) -> None:
    model.eval()
    labels = sample_labels(64, device) if config.conditional else None
    if config.family == "rbm":
        samples = model.sample(64, device, steps=max(args.cd_steps, 50), y=labels)
        samples = from_rbm_range(samples)
    else:
        init = torch.empty(64, 1, 28, 28, device=device).uniform_(-1.0, 1.0)
        samples = sgld_sample(
            model,
            init,
            steps=args.sgld_steps,
            step_size=args.sgld_step_size,
            noise_scale=args.sgld_noise_scale,
            y=labels,
        )
    save_image(unnormalize_to_zero_to_one(samples), out_dir / f"samples_{epoch:03d}.png", nrow=8)
    model.train()


def deep_negative_samples(model, replay: torch.Tensor, labels: torch.Tensor | None, args: argparse.Namespace) -> torch.Tensor:
    return sgld_sample(
        model,
        replay,
        steps=args.sgld_steps,
        step_size=args.sgld_step_size,
        noise_scale=args.sgld_noise_scale,
        y=labels,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir / args.variant)
    device = default_device()
    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)
    model, config = build_mnist_ebm(
        args.variant,
        hidden_dim=args.hidden_dim,
        base_channels=args.base_channels,
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    replay = torch.empty(args.batch_size, 1, 28, 28, device=device).uniform_(-1.0, 1.0)
    rbm_replay = torch.rand(args.batch_size, 1, 28, 28, device=device)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            use_labels = labels if config.conditional else None

            if config.family == "rbm":
                data = to_rbm_range(images)
                if args.persistent:
                    persistent = rbm_replay[: data.shape[0]]
                    loss, negative = pcd_loss(model, data, persistent, steps=args.cd_steps, y=use_labels)
                    rbm_replay[: data.shape[0]] = negative.detach()
                else:
                    loss, _ = cd_loss(model, data, steps=args.cd_steps, y=use_labels)
            elif config.objective == "contrastive-divergence":
                negative = deep_negative_samples(model, replay[: images.shape[0]], use_labels, args)
                loss = contrastive_energy_loss(model, images, negative, use_labels)
                replay[: images.shape[0]] = negative.detach()
            elif config.objective == "joint-energy":
                negative = deep_negative_samples(model, replay[: images.shape[0]], None, args)
                loss = contrastive_energy_loss(model, images, negative, None)
                loss = loss + args.classifier_weight * F.cross_entropy(model.logits(images), labels)
                replay[: images.shape[0]] = negative.detach()
            elif config.objective == "denoising-score-matching":
                loss = denoising_score_matching_loss(model, images, sigma=args.score_sigma)
            elif config.objective == "sliced-score-matching":
                loss = sliced_score_matching_loss(model, images)
            elif config.objective == "noise-contrastive-estimation":
                noise = torch.empty_like(images).uniform_(-1.0, 1.0)
                loss = nce_loss(model, images, noise)
            else:
                raise ValueError(f"Unsupported EBM objective: {config.objective}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": checkpoint_config(args.variant, args.hidden_dim, args.base_channels),
            },
            out_dir / f"model_{epoch:03d}.pt",
        )
        print(f"epoch={epoch} loss={running_loss / len(loader):.4f}")

        if epoch % args.sample_every == 0:
            save_samples(model, config, epoch, out_dir, device, args)


if __name__ == "__main__":
    main()
