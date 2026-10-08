"""Pre-trained backbones (ImageNet weights) via timm, fine-tuned for six-class sleep staging."""
from __future__ import annotations

from typing import Tuple

import torch

from ..constants import PRETRAINED_MODELS


def build_pretrained(key: str, num_classes: int = 6, pretrained: bool = True) -> Tuple[torch.nn.Module, tuple, tuple]:
    """Create a timm model with a fresh ``num_classes`` head.

    Returns ``(model, mean, std)`` where mean/std are the normalisation statistics the weights were
    trained with. ``key`` is a thesis model name (see ``constants.PRETRAINED_MODELS``) or a raw timm name.
    """
    import timm

    timm_name = PRETRAINED_MODELS.get(key, key)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    data_cfg = timm.data.resolve_model_data_config(model)
    return model, tuple(data_cfg["mean"]), tuple(data_cfg["std"])


def extract_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pooled pre-logits embedding for any classifier in this project (timm or ScratchViT)."""
    if hasattr(model, "extract_features") and not hasattr(model, "forward_head"):
        return model.extract_features(x)
    feats = model.forward_features(x)
    return model.forward_head(feats, pre_logits=True)
