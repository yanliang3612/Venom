from __future__ import annotations

import random
from pathlib import Path

import torch


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5
