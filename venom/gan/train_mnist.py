from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.gan.factory import GAN_VARIANTS, build_mnist_gan, checkpoint_config
from venom.gan.losses import discriminator_loss, generator_loss, gradient_penalty
from venom.utils import default_device, ensure_dir, seed_everything, unnormalize_to_zero_to_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Venom GAN models on MNIST.")
    parser.add_argument("--variant", choices=GAN_VARIANTS, default="dcgan")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_gan"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--gp-weight", type=float, default=10.0)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--info-weight", type=float, default=1.0)
    parser.add_argument("--clip-value", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_latents(
    batch_size: int,
    latent_dim: int,
    code_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    z = torch.randn(batch_size, latent_dim, device=device)
    code = torch.empty(batch_size, code_dim, device=device).uniform_(-1, 1) if code_dim > 0 else None
    return z, code


@torch.no_grad()
def save_samples(generator, config, epoch: int, out_dir: Path, device: torch.device, latent_dim: int) -> None:
    generator.eval()
    labels = torch.arange(64, device=device) % 10 if config.conditional else None
    z, code = sample_latents(64, latent_dim, config.info_code_dim, device)
    samples = generator(z, labels, code)
    save_image(unnormalize_to_zero_to_one(samples), out_dir / f"samples_{epoch:03d}.png", nrow=8)
    generator.train()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir / args.variant)
    device = default_device()
    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)
    generator, discriminator, config = build_mnist_gan(
        args.variant,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    )
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    betas = (0.0, 0.9) if config.loss in {"wgan", "hinge"} else (0.5, 0.999)
    g_opt = Adam(generator.parameters(), lr=args.lr, betas=betas)
    d_opt = Adam(discriminator.parameters(), lr=args.lr, betas=betas)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        running_g = 0.0
        running_d = 0.0
        seen_batches = 0

        for real, labels in progress:
            real = real.to(device)
            labels = labels.to(device)
            batch_size = real.shape[0]
            y = labels if config.conditional else None

            d_loss_value = 0.0
            for _ in range(config.critic_steps):
                z, code = sample_latents(batch_size, args.latent_dim, config.info_code_dim, device)
                fake = generator(z, y, code).detach()
                real_out = discriminator(real, y)
                fake_out = discriminator(fake, y)
                d_loss = discriminator_loss(config.loss, real_out["logits"], fake_out["logits"])

                if config.auxiliary_classifier:
                    d_loss = d_loss + args.aux_weight * F.cross_entropy(real_out["class_logits"], labels)
                if config.info_code_dim > 0 and code is not None:
                    d_loss = d_loss + args.info_weight * F.mse_loss(fake_out["code"], code)
                if config.use_gradient_penalty:
                    d_loss = d_loss + args.gp_weight * gradient_penalty(discriminator, real, fake, y)

                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()

                if config.use_weight_clipping:
                    for param in discriminator.parameters():
                        param.data.clamp_(-args.clip_value, args.clip_value)
                d_loss_value += d_loss.item()

            z, code = sample_latents(batch_size, args.latent_dim, config.info_code_dim, device)
            fake = generator(z, y, code)
            fake_out = discriminator(fake, y)
            g_loss = generator_loss(config.loss, fake_out["logits"])

            if config.auxiliary_classifier:
                g_loss = g_loss + args.aux_weight * F.cross_entropy(fake_out["class_logits"], labels)
            if config.info_code_dim > 0 and code is not None:
                g_loss = g_loss + args.info_weight * F.mse_loss(fake_out["code"], code)

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            global_step += 1
            seen_batches += 1
            running_d += d_loss_value / config.critic_steps
            running_g += g_loss.item()
            progress.set_postfix(d=f"{running_d / seen_batches:.4f}", g=f"{running_g / seen_batches:.4f}")

        checkpoint = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_opt.state_dict(),
            "d_optimizer": d_opt.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": checkpoint_config(args.variant, args.latent_dim, args.base_channels),
        }
        torch.save(checkpoint, out_dir / f"model_{epoch:03d}.pt")
        print(f"epoch={epoch} d_loss={running_d / len(loader):.4f} g_loss={running_g / len(loader):.4f}")

        if epoch % args.sample_every == 0:
            save_samples(generator, config, epoch, out_dir, device, args.latent_dim)


if __name__ == "__main__":
    main()
