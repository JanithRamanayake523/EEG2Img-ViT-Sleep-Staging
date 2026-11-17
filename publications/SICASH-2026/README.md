<div align="center">

# Systematic Evaluation of Signal-to-Image Transformation Pipelines for Vision Transformer-based Sleep Stage Classification

Janith R.R.H. Ramanayakage · [Chandima N.P.G. Arachchige](https://www.res.cmb.ac.lk/statistics/chandima-priyadarshani/)

*Department of Statistics, Faculty of Science, University of Colombo, Sri Lanka*

![License](https://img.shields.io/badge/license-MIT-green)
![Dataset](https://img.shields.io/badge/dataset-CAP%20Sleep%20Database-blue)
![Model](https://img.shields.io/badge/model-ViT--B%2F16-6A2C91)

[Slides (PDF)](SICASH%20Presentation.pdf) · [Slides (PPTX)](SICASH%20Presentation.pptx) · [Code](../../)

</div>

---

## Highlights

- Compares **four signal-to-image encodings** (GASF, GADF, MTF, spectrogram) as input to Vision Transformers for six-stage sleep scoring.
- Finds that **GASF + DWT resampling to 128 Hz + a fine-tuned ImageNet ViT-B/16** is the best pipeline, reaching **86.2% accuracy** and **0.853 macro-F1**.
- Evaluated with **subject-wise 5-fold cross-validation** on 80 CAP recordings (80,667 epochs), so no subject appears in both training and test data.
- Class imbalance handled with **DCGAN synthetic augmentation applied to training folds only**.

## Background and motivation

This paper comes from research carried out in 2025 as the final year research project for the **BSc (Hons) in Data Science** degree at the **Department of Statistics, Faculty of Science, University of Colombo**. The research was supervised by **[Dr. Chandima N.P.G. Arachchige](https://www.res.cmb.ac.lk/statistics/chandima-priyadarshani/)**, and the paper presents part of that project.

Sleep disorders such as insomnia, sleep apnoea and narcolepsy affect tens of millions of adults and are linked to reduced quality of life, impaired cognition and higher cardiovascular and metabolic risk. Diagnosis and treatment follow-up depend on **sleep stage classification**, which is still done by trained technologists manually scoring polysomnography (PSG) recordings. This is labour-intensive, taking roughly 1–3 hours per full-night recording, and subjective: agreement even among experienced scorers is only about 82.6%.

Deep learning can automate this task, and Vision Transformers (ViTs) have shown strong results in image recognition. Applying them to PSG requires first converting one-dimensional signals into two-dimensional images, yet existing ViT-based sleep studies mostly use spectrograms or raw signals. It was unclear which encoding suits ViTs best, which sampling rate and resampling method to use before encoding, and how much ImageNet pre-training helps on PSG-derived images. This research answers those questions through controlled experiments on a clinically heterogeneous dataset, the CAP Sleep Database, which includes healthy subjects and patients with sleep disorders.

## Abstract

Sleep stage classification is a critical prerequisite for diagnosing sleep disorders and assessing sleep quality. While Vision Transformers (ViTs) have demonstrated remarkable success in image recognition, existing ViT-based approaches to polysomnography (PSG) sleep staging rely mainly on spectrogram or raw-signal inputs, and the optimal transformation of one-dimensional physiological signals into image representations suitable for ViT processing has not been systematically investigated. This paper presents a comprehensive evaluation of four image transformation techniques: Gramian Angular Summation Field (GASF), Gramian Angular Difference Field (GADF), Markov Transition Field (MTF), and spectrogram, for ViT-based sleep stage classification using the CAP Sleep Database (80 recordings, 80,667 epochs) with subject-wise 5-fold cross-validation. Experimental results demonstrate that GASF achieves the highest classification accuracy at 78.5% among the four transformations. Furthermore, Discrete Wavelet Transform (DWT) down sampling at 128 Hz optimizes performance, yielding 80.3% accuracy. Fine-tuning an ImageNet pre-trained ViT-B/16 model on balanced GASF representations (class balance obtained through DCGAN-based synthetic augmentation) achieves 86.2% accuracy (macro-F1 0.853), representing an improvement of 11.4 percentage points over training from scratch and 10.0 percentage points over a ResNet-50 baseline. These results indicate that GASF encoding with DWT resampling to 128 Hz is an effective preprocessing pipeline for ViT-based sleep stage classification, and that transfer learning from natural image pre-training is highly effective for PSG-derived representations.

**Keywords:** Sleep stage classification; Vision Transformer; Gramian Angular Field; Signal-to-image transformation; Transfer learning

## Citation

If you use this work, please cite:

```bibtex
@misc{ramanayakage2026signal2image,
  title     = {Systematic Evaluation of Signal-to-Image Transformation Pipelines for Vision Transformer-based Sleep Stage Classification},
  author    = {Ramanayakage, Janith R.R.H. and Arachchige, Chandima N.P.G.},
  year      = {2026}
}
```

## License

Released under the [MIT License](../../LICENSE).

## Contact

Janith R.R.H. Ramanayakage: janithramanayake523@gmail.com
