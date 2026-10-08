"""Fixed definitions taken from the thesis (Chapters 4 and 6)."""

# --------------------------------------------------------------------------- sleep stages
# R&K scoring used by the CAP Sleep Database. Movement time (MT) is merged into Wake,
# exactly as in the original notebooks.
STAGE_TO_ID = {"W": 0, "MT": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "R": 5, "REM": 5}
STAGE_NAMES = ["W", "S1", "S2", "S3", "S4", "REM"]
NUM_CLASSES = len(STAGE_NAMES)

# --------------------------------------------------------------------------- subject groups
# File-name prefix -> group name. Subject ids look like n1, ins8, brux2, narco4, nfle12 ...
GROUP_PREFIXES = {
    "n": "healthy",
    "ins": "insomnia",
    "brux": "bruxism",
    "narco": "narcolepsy",
    "nfle": "nfle",
    "plm": "plm",
    "rbd": "rbd",
    "sdb": "sdb",
}

# Data subsets of Section 4.2. DS9 is the union of everything.
DATASET_GROUPS = {
    "DS1": ["healthy"],
    "DS2": ["insomnia"],
    "DS3": ["bruxism"],
    "DS4": ["narcolepsy"],
    "DS5": ["nfle"],
    "DS6": ["plm"],
    "DS7": ["rbd"],
    "DS8": ["sdb"],
    "DS9": list(GROUP_PREFIXES.values()),
}

# --------------------------------------------------------------------------- channels
# The 16 channels that form the 4x4 image grid (thesis Section 6.2.2).
CANONICAL_CHANNELS = [
    "F7-T3", "T3-T5", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "F8-T4", "T4-T6",
    "EOG", "EMG", "ECG", "Avg",
]
EEG_CHANNELS = CANONICAL_CHANNELS[:12]
SOURCE_CHANNELS = CANONICAL_CHANNELS[:-1]  # everything except the computed average

# Names under which each canonical channel appears in the EDF files (matched case-insensitively
# after stripping an "EEG " prefix). Several CAP recordings label Fp1/Fp2 as F1/F2.
CHANNEL_ALIASES = {
    "F7-T3": ["F7-T3"],
    "T3-T5": ["T3-T5"],
    "Fp1-F3": ["Fp1-F3", "F1-F3"],
    "F3-C3": ["F3-C3"],
    "C3-P3": ["C3-P3"],
    "P3-O1": ["P3-O1"],
    "Fp2-F4": ["Fp2-F4", "F2-F4"],
    "F4-C4": ["F4-C4"],
    "C4-P4": ["C4-P4"],
    "P4-O2": ["P4-O2"],
    "F8-T4": ["F8-T4"],
    "T4-T6": ["T4-T6"],
    "EOG": ["ROC-LOC", "LOC-ROC", "EOG dx", "EOG sx"],
    "EMG": ["EMG1-EMG2"],
    "ECG": ["ECG1-ECG2", "ekg", "ecg"],
}

# When a recording does not contain a channel, it is filled from a neighbouring/contralateral
# derivation (this is what the per-subject notebooks did with `channel_mapping`).
CHANNEL_FALLBACKS = {
    "F7-T3": ["F3-C3", "F4-C4"],
    "T3-T5": ["C3-P3", "C4-P4"],
    "Fp1-F3": ["Fp2-F4"],
    "F3-C3": ["F4-C4"],
    "C3-P3": ["C4-P4"],
    "P3-O1": ["P4-O2"],
    "Fp2-F4": ["Fp1-F3", "F1-F3"],
    "F4-C4": ["F3-C3"],
    "C4-P4": ["C3-P3"],
    "P4-O2": ["P3-O1"],
    "F8-T4": ["F4-C4", "F3-C3"],
    "T4-T6": ["C4-P4", "C3-P3"],
    "EOG": [],
    "EMG": ["DX1-DX2", "SX1-SX2", "tib dx", "tib sin"],
    "ECG": [],
}

# --------------------------------------------------------------------------- filters (Table 6.1)
# (high-pass cutoff, low-pass cutoff) in Hz, plus a 60 Hz notch for every channel.
FILTER_BANDS = {
    "eeg": (0.5, 35.0),
    "eog": (0.3, 35.0),
    "ecg": (0.3, 70.0),
    "emg": (10.0, 70.0),
}
NOTCH_HZ = 60.0

# --------------------------------------------------------------------------- image settings
IMAGE_SIZE = 224
GRID_SIZE = 4  # 16 channels -> 4x4 tiles
TRANSFORMS = ("gasf", "gadf", "mtf", "spectrogram")

# --------------------------------------------------------------------------- pre-trained models
# Thesis name -> timm model name. PVT-Small is not in recent timm releases; pvt_v2_b2 is the
# closest size-matched PVT variant.
PRETRAINED_MODELS = {
    "vit_b16": "vit_base_patch16_224",
    "vit_b32": "vit_base_patch32_224",
    "deit_b": "deit_base_patch16_224",
    "swin_b": "swin_base_patch4_window7_224",
    "beit_b": "beit_base_patch16_224",
    "pvt_small": "pvt_v2_b2",
}
