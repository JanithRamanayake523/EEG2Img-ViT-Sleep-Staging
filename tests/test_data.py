import numpy as np
import pandas as pd
import pytest

from sleepvit.config import load_config
from sleepvit.data.annotations import hypnogram_to_epoch_labels, parse_hypnogram
from sleepvit.data.discovery import parse_subject_id
from sleepvit.data.edf_preprocess import ChannelResolutionError, epoch_signal, resolve_channels, standardize
from sleepvit.data.manifest import balanced_sample, split, upsample
from sleepvit.data.resampling import resample_dwt, resample_fft

HYPNO = """RemLogic Event Export
Patient:\tN 3

Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\tUnknown Position\t23:59:00\tSLEEP-S0\t30\tROC-LOC
W\tUnknown Position\t23:59:30\tSLEEP-S0\t30\tROC-LOC
S1\tUnknown Position\t23:59:45\tMCAP-A3\t3\tEEG-Fp2-F4
S1\tUnknown Position\t00:00:00\tSLEEP-S1\t30\tROC-LOC
S2\tUnknown Position\t00:00:30\tSLEEP-S2\t30\tROC-LOC
R\tUnknown Position\t00:01:30\tSLEEP-REM\t30\tROC-LOC
"""


def test_hypnogram_parsing_handles_midnight_and_gaps(tmp_path):
    f = tmp_path / "n3.txt"
    f.write_text(HYPNO)
    df = parse_hypnogram(f)
    assert df["onset_s"].is_monotonic_increasing
    labels = hypnogram_to_epoch_labels(df, 30)
    # epochs: 0=W 1=W 2=S1 3=S2 4=(gap) 5=REM
    assert labels.tolist() == [0, 0, 1, 2, -1, 5]


def test_subject_id_parsing():
    assert parse_subject_id("nfle12") == ("nfle12", "nfle")
    assert parse_subject_id("n1") == ("n1", "healthy")
    assert parse_subject_id("brux2") == ("brux2", "bruxism")
    assert parse_subject_id("SC4001E0-PSG") is None


def test_channel_resolution_with_fallbacks():
    available = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                 "ROC-LOC", "EMG1-EMG2", "ECG1-ECG2"]
    src = resolve_channels(available)
    assert src["F7-T3"] == "F3-C3"        # missing -> falls back to a neighbouring derivation
    assert src["Fp1-F3"] == "FP1-F3"       # case-insensitive alias
    with pytest.raises(ChannelResolutionError):
        resolve_channels(["F3-C3"])


def test_dwt_and_fft_resampling_lengths():
    x = np.random.randn(3, 512 * 60)
    for target in (256, 128, 64):
        assert resample_dwt(x, 512, target).shape[-1] == 512 * 60 * target // 512
        assert resample_fft(x, 512, target).shape[-1] == 512 * 60 * target // 512
    with pytest.raises(ValueError):
        resample_dwt(x, 512, 100)


def test_epoching_and_standardisation():
    data = np.random.randn(16, 128 * 95)
    ep = epoch_signal(data, 128, 30)
    assert ep.shape == (3, 16, 3840)
    z = standardize(ep)
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-6)
    assert np.allclose(z.std(axis=(0, 2)), 1, atol=1e-6)


def _toy_manifest():
    rows = [{"subject": f"s{i % 4}", "epoch": i, "stage": i % 3, "path": f"{i}.png"} for i in range(120)]
    return pd.DataFrame(rows)


def test_splits_and_balancing():
    df = _toy_manifest()
    train, val = split(df, 0.25, seed=1, by="subject")
    assert set(train["subject"]).isdisjoint(val["subject"])
    train, val = split(df, 0.25, seed=1, by="epoch")
    assert len(train) + len(val) == len(df)
    skewed = pd.concat([df, df[df["stage"] == 0].assign(epoch=lambda d: d["epoch"] + 1000)])
    assert upsample(skewed)["stage"].value_counts().nunique() == 1
    assert balanced_sample(df, 10)["stage"].value_counts().eq(10).all()


def test_config_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("a:\n  b: 1\nlist: [1, 2]\n")
    cfg = load_config(f, ["a.b=5", "a.c=true"])
    assert cfg.a.b == 5 and cfg.a.c is True
