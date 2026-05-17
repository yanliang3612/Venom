from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from venom.data import make_mnist_loader
from venom.diffusion.factory import build_mnist_diffusion, checkpoint_config
from venom.diffusion.one_step import ProgressiveDistillation
from venom.utils import default_device, ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a DDPM-family MNIST teacher into a student.")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_diffusion/progressive-distillation"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--student-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = default_device()

    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    config = checkpoint["config"]
    teacher_model, teacher = build_mnist_diffusion(
        config["variant"],
        timesteps=config.get("timesteps", 1000),
        base_channels=config.get("base_channels", 64),
        class_dropout=config.get("class_dropout", 0.0),
        backbone=config.get("backbone", "unet"),
    )
    teacher_model.load_state_dict(checkpoint["model"])
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student_model, student = build_mnist_diffusion(
        config["variant"],
        timesteps=config.get("timesteps", 1000),
        base_channels=config.get("base_channels", 64),
        class_dropout=config.get("class_dropout", 0.0),
        backbone=config.get("backbone", "unet"),
    )
    student = student.to(device)
    distiller = ProgressiveDistillation(student, teacher, student_steps=args.student_steps)
    optimizer = AdamW(student.parameters(), lr=args.lr)
    loader = make_mnist_loader(args.data_dir, args.batch_size, args.num_workers)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"progressive distillation epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            use_labels = labels if config["variant"] in {"adm", "cfg"} else None
            loss = distiller.training_loss(images, use_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        save_config = checkpoint_config(
            config["variant"],
            config.get("timesteps", 1000),
            config.get("base_channels", 64),
            config.get("class_dropout", 0.0),
            config.get("backbone", "unet"),
        )
        save_config["distilled_steps"] = args.student_steps
        torch.save(
            {
                "model": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": save_config,
            },
            out_dir / f"student_{epoch:03d}.pt",
        )
        print(f"epoch={epoch} loss={running_loss / len(loader):.4f}")


if __name__ == "__main__":
    main()
