"""Convert recordings to labelled 30 s epoch images.

Each *setting* is ``<transform>_<fs>hz_<resample>``, e.g. ``gasf_128hz_dwt`` or ``gadf_512hz_orig``.

    # Exp. 1: four transformations at the original 512 Hz
    python scripts/01_build_images.py --transforms gasf gadf mtf spectrogram --fs 512 --resample orig
    # Exp. 2: sampling frequency / resampling method grid for GASF
    python scripts/01_build_images.py --transforms gasf --fs 256 128 64 --resample fft dwt
    # a single named setting, restricted to a few subjects
    python scripts/01_build_images.py --settings gasf_128hz_dwt --subjects n1 ins8 --workers 4
"""
import argparse
import itertools

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import TRANSFORMS
from sleepvit.data.build_images import build_dataset, parse_setting
from sleepvit.data.discovery import discover_subjects
from sleepvit.utils import get_logger, seed_everything

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--settings", nargs="*", help="Named settings such as gasf_128hz_dwt")
    parser.add_argument("--transforms", nargs="*", choices=TRANSFORMS)
    parser.add_argument("--fs", nargs="*", type=int, default=[512])
    parser.add_argument("--resample", nargs="*", default=["orig"], choices=["orig", "fft", "dwt"])
    parser.add_argument("--subjects", nargs="*", help="Only these subject ids")
    parser.add_argument("--workers", type=int, default=None, help="Parallel subjects (default: images.workers)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)

    settings = [parse_setting(s) for s in (args.settings or [])]
    if args.transforms:
        for t, fs, m in itertools.product(args.transforms, args.fs, args.resample):
            if m == "orig" and fs != cfg.data.required_fs:
                continue  # 'orig' only makes sense at the native rate
            if m != "orig" and fs == cfg.data.required_fs:
                continue
            settings.append((t, fs, m))
    if not settings:
        parser.error("give --settings or --transforms")
    settings = sorted(set(settings))
    log.info("settings: %s", settings)

    subjects = discover_subjects(cfg.paths.raw_data, required_fs=cfg.data.required_fs)
    if args.subjects:
        subjects = [s for s in subjects if s.sid in set(args.subjects)]
    log.info("%d subjects selected", len(subjects))
    build_dataset(subjects, settings, cfg, cfg.paths.processed,
                  workers=args.workers or cfg.images.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
