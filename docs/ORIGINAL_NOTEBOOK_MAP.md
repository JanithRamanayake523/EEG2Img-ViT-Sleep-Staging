# Where each original notebook went

| Original (`UG Research/...`) | What it did | Now |
|---|---|---|
| `Research/Code/n/n1.ipynb`, `n3`, `n5`, `n10`, `n11`, `ins/ins8`, `ins9`, `brux/brux1`, `brux2`, ... | One near-identical notebook per subject: load EDF, patch missing channels, add `Average`, filter, epoch, z-score, make GASF/GADF/spectrogram/MTF images | `data/edf_preprocess.py` (`resolve_channels`, `load_recording`), `data/image_transforms.py`, `data/build_images.py`, `scripts/01_build_images.py`. One code path for every subject. |
| `Research/Code/stages.ipynb` (and `*_stages.csv`) | Parse RemLogic `.txt` hypnograms, shift clock times by hand, keep 30 s rows, map stages to 0-5 | `data/annotations.py`; labels are computed per subject on the fly and stored in the manifest CSV. |
| `Research/Classify/classifycode.ipynb` | Move PNGs into `0..5` folders and then `c0..c5` | Not needed: images stay in per-subject folders and the manifest (`data/processed/manifests/<setting>.csv`) holds `image, subject, group, epoch, stage`. |
| `Research/Code/CreateDS.ipynb` | Train/val split, small CNN, Keras ViT | `data/manifest.py`, `data/datasets.py`, `models/vit_scratch.py` (PyTorch, Table 7.4 architecture). |
| `Research/ImgType/*imgtype.ipynb` | Compare image types (MobileNetV2 stand-in), t-SNE + silhouette/DBI/CHI | `scripts/02_exp1_transforms.py`, `evaluation/clustering.py`. |
| `Research/Gan/gan.ipynb` | Keras DCGAN on one class folder | `models/dcgan.py` (Tables 7.6/7.7), `training/gan_trainer.py`, `scripts/06_exp5_train_dcgan.py`, `scripts/07_exp6_gan_augmentation.py`. |
| `Research/Code/Spectrogram *.ipynb` | Spectrogram variants | `spectrogram` transform in `data/image_transforms.py` |
| `Research/Temp/temp.ipynb` | Plots for the thesis | `evaluation/plots.py`, `scripts/09_make_report.py` (see note 6 in `NOTES.md`) |
| `EEG2Img Work/Edf codes/*.ipynb`, `EEG2Img Work/test.ipynb` | Sleep-EDF and BCI-competition side experiments | Out of scope for the thesis; not ported. |
