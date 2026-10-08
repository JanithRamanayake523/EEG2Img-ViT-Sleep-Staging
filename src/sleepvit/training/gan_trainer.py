"""DCGAN training, per-class fine-tuning and synthetic image generation (Experiments 5 and 6)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.utils import save_image

from ..data.datasets import gan_transform, make_loader
from ..models.dcgan import Discriminator, Generator
from ..utils import ensure_dir, get_logger

log = get_logger()


def build_gan(cfg, device):
    g = Generator(cfg.gan.latent_dim).to(device)
    d = Discriminator().to(device)
    return g, d


def train_gan(cfg, df: pd.DataFrame, device: torch.device, out_dir: str | Path, epochs: int,
              init_from: Optional[str | Path] = None, tag: str = "gan"):
    """Train (or fine-tune from ``init_from``) a DCGAN on the images listed in ``df``.

    Non-saturating GAN loss with Adam(lr, beta1=0.5). Label smoothing on the real target keeps the
    discriminator from becoming over-confident. Sample grids and checkpoints are written to ``out_dir``.
    """
    c = cfg.gan
    out_dir = ensure_dir(out_dir)
    g, d = build_gan(cfg, device)
    if init_from:
        state = torch.load(init_from, map_location=device)
        g.load_state_dict(state["generator"])
        d.load_state_dict(state["discriminator"])
        log.info("initialised from %s", init_from)

    loader = make_loader(df, gan_transform(cfg.images.size), c.batch_size, True, c.num_workers, drop_last=True)
    opt_g = torch.optim.Adam(g.parameters(), lr=c.lr_g, betas=(c.beta1, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=c.lr_d, betas=(c.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(16, c.latent_dim, device=device)
    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        d_sum = g_sum = 0.0
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            bsz = real.size(0)
            noise = torch.randn(bsz, c.latent_dim, device=device)
            fake = g(noise)

            # discriminator step: real -> label_smoothing, fake -> 0
            opt_d.zero_grad(set_to_none=True)
            loss_d = bce(d(real), torch.full((bsz,), c.label_smoothing, device=device)) + \
                bce(d(fake.detach()), torch.zeros(bsz, device=device))
            loss_d.backward()
            opt_d.step()

            # generator step: make the discriminator call fakes real
            opt_g.zero_grad(set_to_none=True)
            loss_g = bce(d(fake), torch.ones(bsz, device=device))
            loss_g.backward()
            opt_g.step()
            d_sum += loss_d.item()
            g_sum += loss_g.item()
        history["d_loss"].append(d_sum / len(loader))
        history["g_loss"].append(g_sum / len(loader))
        if epoch % 10 == 0 or epoch == 1:
            log.info("[%s] epoch %d/%d | D %.3f | G %.3f", tag, epoch, epochs, history["d_loss"][-1], history["g_loss"][-1])
        if epoch % c.sample_every == 0 or epoch == epochs:
            g.eval()
            with torch.no_grad():
                save_image(g(fixed_noise) * 0.5 + 0.5, Path(out_dir) / f"samples_epoch{epoch:04d}.png", nrow=4)
            g.train()
        if epoch % c.checkpoint_every == 0 or epoch == epochs:
            torch.save({"generator": g.state_dict(), "discriminator": d.state_dict(), "epoch": epoch},
                       Path(out_dir) / "gan.pt")
    return g, d, history


def load_generator(cfg, checkpoint: str | Path, device: torch.device) -> Generator:
    g = Generator(cfg.gan.latent_dim).to(device)
    g.load_state_dict(torch.load(checkpoint, map_location=device)["generator"])
    return g.eval()


@torch.no_grad()
def generate_images(g: Generator, n: int, out_dir: str | Path, latent_dim: int, device: torch.device,
                    batch_size: int = 64, prefix: str = "syn") -> None:
    """Sample ``n`` images from the generator and save them as PNG files."""
    out_dir = ensure_dir(out_dir)
    g.eval()
    written = 0
    while written < n:
        b = min(batch_size, n - written)
        imgs = g(torch.randn(b, latent_dim, device=device)) * 0.5 + 0.5
        arr = (imgs.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for k, a in enumerate(arr):
            Image.fromarray(a).save(Path(out_dir) / f"{prefix}_{written + k:06d}.png")
        written += b
