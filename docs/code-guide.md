# sleepvit - Sleep stage classification with Vision Transformers

Code base for the undergraduate research *Enhancing Sleep Stage Classification with Vision Transformers: A Study
on Image Transformations, Data Augmentation, and Model Optimization* (University of Colombo, 2025).

PSG epochs from the PhysioNet **CAP Sleep Database** are converted to images (GASF, GADF, MTF, spectrogram),
down-sampled with FFT or DWT, optionally augmented with a DCGAN, and classified into six R&K stages
(W, S1, S2, S3, S4, REM) by Vision Transformers.

> This code base was assembled from the research notebooks and the thesis, but **has not been executed yet**.
> Read [NOTES.md](NOTES.md) before running: it lists deviations from the original notebooks (including
> a filter bug) and methodology points that affect the reported numbers.

## Pipeline

```
 EDF + hypnogram .txt ──► load / harmonise 16 channels ──► filter ──► (FFT | DWT) resample ──► 30 s epochs
        ──► z-score ──► transform per channel ──► 4x4 tile ──► 224x224 PNG + manifest CSV
        ──► ViT (scratch | pre-trained)  ◄── DCGAN synthetic images (per sleep stage)
```

The 16 channels are F7-T3, T3-T5, Fp1-F3, F3-C3, C3-P3, P3-O1, Fp2-F4, F4-C4, C4-P4, P4-O2, F8-T4, T4-T6,
EOG, EMG, ECG and the channel average.

## Layout

```
configs/default.yaml        every setting (paths, filters, image params, training, GAN)
src/sleepvit/
  constants.py              stages, groups DS1-DS9, channel aliases/fallbacks, filter bands, model names
  config.py, utils.py
  data/                     discovery, annotations (hypnograms), edf_preprocess, resampling (FFT/DWT),
                            image_transforms, build_images, manifest (subsets/splits), datasets
  models/                   vit_scratch (Table 7.4), pretrained (timm), dcgan (Tables 7.6/7.7)
  training/                 trainer, pipeline (one run), gan_trainer, metrics
  evaluation/               clustering (t-SNE, silhouette, DBI, CHI), gan_metrics (SSIM, FID, MPED), plots
  experiment_utils.py       identical epoch samples/splits across settings
scripts/                    one script per pipeline stage / thesis experiment
tests/                      unit tests for parsing, resampling, shapes, splits
docs/                       NOTES.md, ORIGINAL_NOTEBOOK_MAP.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                       # or: pip install -r requirements.txt
pytest                                                  # optional sanity check
```

Download the CAP Sleep Database (https://physionet.org/content/capslpdb/1.0.0/) and put the `.edf` files and their
`.txt` hypnograms anywhere under `data/raw/` (the folders are searched recursively; only the file names matter:
`n1.edf`, `n1.txt`, `ins8.edf`, ...). To use the data already on disk, point the config at it:

```bash
python scripts/00_inspect_dataset.py -o paths.raw_data="../Research/Data"
```

## Reproducing the experiments

All scripts accept `--config configs/default.yaml` and repeated `-o key=value` overrides.

| Step | Command | Thesis |
|---|---|---|
| Inventory of subjects and stage counts | `python scripts/00_inspect_dataset.py` | Tables 4.1, 4.2 |
| Primary analysis | `python scripts/10_primary_analysis.py --artifact-subject n1 --subjects n1 ins8 --stage S3` | Chapter 5 |
| Images for Exp. 1 | `python scripts/01_build_images.py --transforms gasf gadf mtf spectrogram --fs 512 --resample orig --workers 4` | 6.2.2 |
| Exp. 1 transformations | `python scripts/02_exp1_transforms.py` | Tables 7.1, 7.2 |
| Images for Exp. 2 | `python scripts/01_build_images.py --transforms gasf --fs 256 128 64 --resample fft dwt` | 6.2.3 |
| Exp. 2 sampling / resampling | `python scripts/03_exp2_sampling.py` | Table 7.3 |
| Exp. 3 scratch vs pre-trained | `python scripts/04_exp3_scratch_vs_pretrained.py` | Figure 7.5 |
| Exp. 4 pre-trained models | `python scripts/05_exp4_pretrained_models.py` | Table 7.5 |
| Exp. 5 DCGAN | `python scripts/06_exp5_train_dcgan.py` | Tables 7.6, 7.7, Fig. 7.7 |
| Exp. 6 GAN augmentation | `python scripts/07_exp6_gan_augmentation.py --generate` | Figure 7.8 |
| Final model, DS1-DS8 | `python scripts/08_final_evaluation.py` | Fig. 7.9, Table 7.8 |
| Collect tables | `python scripts/09_make_report.py` | - |

Exp. 3-6 use `best_setting` in the config (GASF, DWT, 128 Hz, the thesis outcome of Exp. 1-2); change it if your
re-run picks a different winner. For a quick smoke test try
`-o experiments.epochs_per_class=50 -o classifier.epochs=1 -o gan.epochs=2 -o gan.pretrain_epochs=1`.

Outputs: images and manifests in `data/processed/`, metrics/tables/plots in `results/`, weights in `checkpoints/`.

## Thesis results for reference (not reproduced here)

GASF 57.2 % > MTF 57.1 > GADF 56.7 > spectrogram 55.4 (Exp. 1); DWT at 128 Hz 60.1 % (Exp. 2); pre-trained ViT-B/16
78.1 % vs from scratch 68.9 % on balanced data (Exp. 3); ViT-B/16 best of six backbones (Exp. 4); mixed
original + synthetic 85.1 % vs original 78.7 % vs synthetic 76.8 % (Exp. 6); final model 84.8 %, 81.0-92.8 % across
DS1-DS8.
