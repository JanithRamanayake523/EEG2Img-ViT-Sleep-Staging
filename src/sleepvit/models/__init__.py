"""Model definitions: from-scratch ViT, pre-trained backbones (timm) and the DCGAN."""
from .dcgan import Discriminator, Generator, weights_init
from .pretrained import build_pretrained
from .vit_scratch import ScratchViT

__all__ = ["Discriminator", "Generator", "weights_init", "build_pretrained", "ScratchViT", "build_classifier"]


def build_classifier(cfg, model_key: str, pretrained: bool = True, num_classes: int = 6):
    """Return ``(model, mean, std)``.

    ``model_key == "scratch"`` builds the from-scratch ViT of Table 7.4; any other key is looked up in
    ``constants.PRETRAINED_MODELS``.
    """
    if model_key == "scratch":
        s = cfg.scratch_vit
        model = ScratchViT(
            image_size=cfg.images.size, patch_size=s.patch_size, num_classes=num_classes,
            embed_dim=s.embed_dim, depth=s.depth, heads=s.heads, mlp_dim=s.mlp_dim,
            dropout=s.dropout, pos_embed=s.pos_embed,
        )
        return model, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return build_pretrained(model_key, num_classes=num_classes, pretrained=pretrained)
