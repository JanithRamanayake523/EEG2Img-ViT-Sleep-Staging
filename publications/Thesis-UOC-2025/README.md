<div align="center">

# Enhancing Sleep Stage Classification with Vision Transformers: A Study on Image Transformations, Data Augmentation, and Model Optimization

Janith R.R.H. Ramanayakage · Supervisor: [Dr. Chandima N.P.G. Arachchige](https://www.res.cmb.ac.lk/statistics/chandima-priyadarshani/)

*BSc (Hons) in Data Science, Department of Statistics, Faculty of Science, University of Colombo, Sri Lanka, March 2025*

![License](https://img.shields.io/badge/license-MIT-green)
![Dataset](https://img.shields.io/badge/dataset-CAP%20Sleep%20Database-blue)
![Model](https://img.shields.io/badge/model-ViT--B%2F16-6A2C91)

[Department of Statistics, UoC](https://science.cmb.ac.lk/statistics/) · [Defence slides (PPTX)](Thesis%20Defence.pptx) · [Code](../../)

</div>

---

## Highlights

- Compares four signal-to-image encodings (**spectrogram, GASF, GADF, MTF**) as input to Vision Transformers for sleep stage classification.
- Examines the effect of **sampling frequency and resampling method**; GASF with **DWT at 128 Hz** performs best.
- Uses a **DCGAN** to generate synthetic images for under-represented sleep stages and measures the effect on accuracy.
- Compares ViTs **trained from scratch against pre-trained ViTs**, and several pre-trained architectures. The final model, **ViT-B/16 fine-tuned on augmented data**, reaches **84.8% accuracy**.

## Background and motivation

This thesis is the final year research project for the **BSc (Hons) in Data Science** degree at the **Department of Statistics, Faculty of Science, University of Colombo**, supervised by **[Dr. Chandima N.P.G. Arachchige](https://www.res.cmb.ac.lk/statistics/chandima-priyadarshani/)**. It is the most detailed account of the project and the starting point for the later ICDS 2025 and SICASH 2026 papers.

The thesis used a subsample of about 12,000 epochs with random epoch-level splits. Adjacent epochs from the same night are highly correlated, so these numbers are optimistic compared with predicting stages for an unseen person. The SICASH 2026 paper re-ran the pipeline with subject-wise cross-validation, so its numbers are the more rigorous ones. See [docs/NOTES.md](../../docs/NOTES.md).

## Abstract

Sleep stage classification is essential for diagnosing sleep disorders and analyzing sleep patterns. This study investigates the effectiveness of Vision Transformers (ViTs) for sleep stage classification using EEG-based image transformations. Multiple transformation techniques, including Spectrograms, Gramian Angular Fields (GADF, GASF), and Markov Transition Fields (MTF), are evaluated to identify the most suitable representation. Additionally, the impact of different sampling frequencies and resampling methods is examined.

To mitigate class imbalance and subject variability, a Deep Convolutional Generative Adversarial Network (DCGAN) is employed to generate synthetic sleep stage images. The effectiveness of GAN-augmented data is assessed by comparing classification performance on datasets with and without synthetic samples. Furthermore, we compare ViTs trained from scratch versus pre-trained ViTs and evaluate various pre-trained models to determine the optimal architecture.

Experimental results show that GASF with Discrete Wavelet Transform (DWT) at 128 Hz yields the best classification performance. GAN-based augmentation improves accuracy by enhancing dataset diversity. The final model, ViT-B/16 fine-tuned on augmented data, achieves the highest classification accuracy of 84.8%. These findings highlight the potential of Vision Transformers in advancing automated sleep stage classification.

**Keywords:** Sleep stage classification; Vision Transformers; EEG; Image transformation; GAN; Deep learning; Data augmentation

## Citation

If you use this work, please cite:

```bibtex
@thesis{ramanayakage2025enhancing,
  title  = {Enhancing Sleep Stage Classification with Vision Transformers: A Study on Image Transformations, Data Augmentation, and Model Optimization},
  author = {Ramanayakage, Janith R.R.H.},
  type   = {BSc (Hons) thesis},
  school = {Department of Statistics, University of Colombo},
  year   = {2025}
}
```

## License

Released under the [MIT License](../../LICENSE).

## Contact

Janith R.R.H. Ramanayakage: janithramanayake523@gmail.com
