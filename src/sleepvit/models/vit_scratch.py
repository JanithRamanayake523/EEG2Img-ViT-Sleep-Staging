"""Vision Transformer trained from scratch (thesis Section 6.1.4 and Table 7.4)."""
from __future__ import annotations

import math

import torch
from torch import nn


def sincos_position_embedding(n_tokens: int, dim: int) -> torch.Tensor:
    """PE(pos, 2i) = sin(pos / 10000^(2i/D)), PE(pos, 2i+1) = cos(...)  (Eq. 6.23-6.24)."""
    position = torch.arange(n_tokens, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_tokens, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Pre-norm block: Z += MHSA(LN(Z)); Z += FFN(LN(Z)) with GELU (Eq. 6.30-6.31)."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ScratchViT(nn.Module):
    """Patch embedding -> [CLS] + positions -> L transformer blocks -> linear head on the CLS token."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, num_classes: int = 6,
                 embed_dim: int = 256, depth: int = 6, heads: int = 8, mlp_dim: int = 512,
                 dropout: float = 0.1, pos_embed: str = "sincos"):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        n_patches = (image_size // patch_size) ** 2
        # Conv with kernel = stride = patch size == flatten each patch and apply a shared linear map (Eq. 6.21)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if pos_embed == "sincos":
            self.register_buffer("pos_embed", sincos_position_embedding(n_patches + 1, embed_dim))
        elif pos_embed == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            raise ValueError("pos_embed must be 'sincos' or 'learned'")
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)  # softmax is folded into the cross-entropy loss
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Final-layer CLS embedding (used for the t-SNE / clustering analysis)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)           # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.dropout(torch.cat([cls, x], dim=1) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))
