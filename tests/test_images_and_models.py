import numpy as np
import pytest

from sleepvit.config import load_config

pyts = pytest.importorskip("pyts")
torch = pytest.importorskip("torch")

from sleepvit.data.image_transforms import epoch_to_image  # noqa: E402
from sleepvit.models.dcgan import Discriminator, Generator  # noqa: E402
from sleepvit.models.vit_scratch import ScratchViT  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


@pytest.mark.parametrize("transform", ["gasf", "gadf", "mtf", "spectrogram"])
def test_epoch_image_shape(cfg, transform):
    epoch = np.random.randn(16, 128 * 30).astype(np.float32)
    img = epoch_to_image(epoch, transform, 128, cfg)
    assert img.size == (224, 224) and img.mode == "RGB"


def test_constant_channel_does_not_crash(cfg):
    epoch = np.random.randn(16, 128 * 30).astype(np.float32)
    epoch[3] = 1.0
    assert epoch_to_image(epoch, "gasf", 128, cfg).size == (224, 224)


def test_scratch_vit_forward_and_features():
    for pos in ("sincos", "learned"):
        m = ScratchViT(pos_embed=pos)
        x = torch.randn(2, 3, 224, 224)
        assert m(x).shape == (2, 6)
        assert m.extract_features(x).shape == (2, 256)


def test_dcgan_shapes():
    g, d = Generator(100), Discriminator()
    g.eval(), d.eval()
    with torch.no_grad():
        fake = g(torch.randn(2, 100))
        assert fake.shape == (2, 3, 224, 224)
        assert fake.min() >= -1 and fake.max() <= 1
        assert d(fake).shape == (2,)
