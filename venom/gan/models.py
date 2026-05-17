from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def maybe_spectral_norm(module: nn.Module, use_spectral_norm: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use_spectral_norm else module


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        inner = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, inner, 1)
        self.key = nn.Conv2d(channels, inner, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2)
        attn = torch.bmm(q, k).softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.gamma * out


class MLPGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 28,
        channels: int = 1,
        hidden_dim: int = 256,
        num_classes: int | None = None,
        label_dim: int = 32,
        code_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.label_emb = nn.Embedding(num_classes, label_dim) if num_classes is not None else None
        input_dim = latent_dim + code_dim + (label_dim if num_classes is not None else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, image_size * image_size * channels),
            nn.Tanh(),
        )

    def _input(self, z: torch.Tensor, y: torch.Tensor | None = None, code: torch.Tensor | None = None) -> torch.Tensor:
        pieces = [z]
        if code is not None:
            pieces.append(code)
        if self.label_emb is not None:
            if y is None:
                y = torch.randint(0, self.num_classes, (z.shape[0],), device=z.device)
            pieces.append(self.label_emb(y.long()))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor | None = None,
        code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.net(self._input(z, y, code))
        return x.reshape(z.shape[0], self.channels, self.image_size, self.image_size)


class MLPDiscriminator(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        hidden_dim: int = 256,
        num_classes: int | None = None,
        label_dim: int = 32,
        auxiliary_classifier: bool = False,
        code_dim: int = 0,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.auxiliary_classifier = auxiliary_classifier
        self.code_dim = code_dim
        self.label_emb = nn.Embedding(num_classes, label_dim) if num_classes is not None and not auxiliary_classifier else None
        input_dim = image_size * image_size * channels + (label_dim if self.label_emb is not None else 0)
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, num_classes) if auxiliary_classifier else None
        self.code_head = nn.Linear(hidden_dim, code_dim) if code_dim > 0 else None

    def _input(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        flat = x.flatten(1)
        if self.label_emb is not None:
            if y is None:
                raise ValueError("Conditional discriminator requires labels")
            flat = torch.cat([flat, self.label_emb(y.long())], dim=1)
        return flat

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.features(self._input(x, y))
        out = {"logits": self.adv_head(h).squeeze(1), "features": h}
        if self.class_head is not None:
            out["class_logits"] = self.class_head(h)
        if self.code_head is not None:
            out["code"] = self.code_head(h)
        return out


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 28,
        channels: int = 1,
        base_channels: int = 64,
        num_classes: int | None = None,
        label_dim: int = 32,
        code_dim: int = 0,
        self_attention: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.label_emb = nn.Embedding(num_classes, label_dim) if num_classes is not None else None
        input_dim = latent_dim + code_dim + (label_dim if num_classes is not None else 0)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, base_channels * 4 * 7 * 7),
            nn.BatchNorm1d(base_channels * 4 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        ]
        if self_attention:
            layers.append(SelfAttention2d(base_channels * 2))
        layers.extend(
            [
                nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, channels, 3, padding=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def _input(self, z: torch.Tensor, y: torch.Tensor | None = None, code: torch.Tensor | None = None) -> torch.Tensor:
        pieces = [z]
        if code is not None:
            pieces.append(code)
        if self.label_emb is not None:
            if y is None:
                y = torch.randint(0, self.num_classes, (z.shape[0],), device=z.device)
            pieces.append(self.label_emb(y.long()))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor | None = None,
        code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.fc(self._input(z, y, code)).reshape(z.shape[0], -1, 7, 7)
        return self.net(h)


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        base_channels: int = 64,
        num_classes: int | None = None,
        label_dim: int = 32,
        auxiliary_classifier: bool = False,
        code_dim: int = 0,
        use_spectral_norm: bool = False,
        self_attention: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.auxiliary_classifier = auxiliary_classifier
        self.code_dim = code_dim
        self.label_emb = nn.Embedding(num_classes, image_size * image_size) if num_classes is not None and not auxiliary_classifier else None
        in_channels = channels + (1 if self.label_emb is not None else 0)
        layers: list[nn.Module] = [
            maybe_spectral_norm(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if self_attention:
            layers.append(SelfAttention2d(base_channels * 2))
        layers.extend(
            [
                maybe_spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), use_spectral_norm),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        self.features = nn.Sequential(*layers)
        feature_dim = base_channels * 4 * 7 * 7
        self.adv_head = maybe_spectral_norm(nn.Linear(feature_dim, 1), use_spectral_norm)
        self.class_head = maybe_spectral_norm(nn.Linear(feature_dim, num_classes), use_spectral_norm) if auxiliary_classifier else None
        self.code_head = nn.Linear(feature_dim, code_dim) if code_dim > 0 else None

    def _input(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_emb is None:
            return x
        if y is None:
            raise ValueError("Conditional discriminator requires labels")
        label_map = self.label_emb(y.long()).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, label_map], dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.features(self._input(x, y)).flatten(1)
        out = {"logits": self.adv_head(h).squeeze(1), "features": h}
        if self.class_head is not None:
            out["class_logits"] = self.class_head(h)
        if self.code_head is not None:
            out["code"] = self.code_head(h)
        return out
