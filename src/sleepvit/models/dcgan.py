"""DCGAN generator and discriminator for 224x224 RGB PSG images (thesis Tables 7.6 and 7.7)."""
from __future__ import annotations

import torch
from torch import nn


def weights_init(m: nn.Module) -> None:
    """Standard DCGAN initialisation: N(0, 0.02) for convolutions, N(1, 0.02) for batch-norm scales."""
    name = m.__class__.__name__
    if "Conv" in name or name == "Linear":
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """z (100) -> Dense+reshape (14,14,1024) -> 4 x [ConvT 4x4 s2 + BN + ReLU] -> ConvT 3x3 s1 + tanh -> 224x224x3."""

    def __init__(self, latent_dim: int = 100, base: int = 1024):
        super().__init__()
        self.base = base
        self.project = nn.Linear(latent_dim, 14 * 14 * base, bias=False)
        self.bn0 = nn.BatchNorm2d(base)

        def up(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.up = nn.Sequential(
            up(base, base // 2),        # 14 -> 28
            up(base // 2, base // 4),   # 28 -> 56
            up(base // 4, base // 8),   # 56 -> 112
            up(base // 8, base // 16),  # 112 -> 224
        )
        self.out = nn.Sequential(nn.ConvTranspose2d(base // 16, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.shape[0], self.base, 14, 14)
        x = torch.relu(self.bn0(x))
        return self.out(self.up(x))


class Discriminator(nn.Module):
    """224x224x3 -> 5 x [Conv 4x4 s2 + LeakyReLU(0.2) (+BN after the first)] -> flatten -> dense logit.

    The final sigmoid of Table 7.7 is folded into ``BCEWithLogitsLoss`` for numerical stability.
    """

    def __init__(self, base: int = 64):
        super().__init__()
        chans = [3, base, base * 2, base * 4, base * 8, base * 16]  # 3,64,128,256,512,1024
        layers = []
        for i in range(5):
            layers.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1, bias=i == 0))
            if i > 0:
                layers.append(nn.BatchNorm2d(chans[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*layers)         # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.classifier = nn.Linear(7 * 7 * chans[-1], 1)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1)).squeeze(1)
