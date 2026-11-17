<div align="center">

# Representation Separability of Signal-to-Image Transformations for Sleep Stage Classification Using Pretrained Vision Transformers

Janith R.R.H. Ramanayakage · [Chandima N.P.G. Arachchige](https://www.res.cmb.ac.lk/statistics/chandima-priyadarshani/)

*Department of Statistics, Faculty of Science, University of Colombo, Sri Lanka*

*International Conference in Data Science (ICDS) 2025, Colombo, Sri Lanka (Paper ID 60)*

![License](https://img.shields.io/badge/license-MIT-green)
![Dataset](https://img.shields.io/badge/dataset-CAP%20Sleep%20Database-blue)
![Model](https://img.shields.io/badge/model-ViT--B%2F32-6A2C91)

[Proceedings](https://science.cmb.ac.lk/icds/icds-2025/proceedings/#flipbook-df_1135/78/) · [Slides (PDF)](ICDS%20Presentation.pdf) · [Code](../../)

</div>

---

## Highlights

- Systematically evaluates **1,016 combinations** of signal-to-image methods, fusion strategies and channel selections.
- Ranks them by **embedding separability** (Fisher's Discriminant Ratio, Silhouette, Davies-Bouldin and Calinski-Harabasz indices) from a fine-tuned ViT-B/16.
- **GASF with grid tiling across all channels** gives the most discriminative representations.
- With that representation, **ViT-B/32** reaches **83.1% accuracy** and **79.4% macro-F1** in five-fold stratified cross-validation, with Swin-B and BEiT-B close behind.

## Background and motivation

This paper comes from research carried out in 2025 as the final year research project for the **BSc (Hons) in Data Science** degree at the **Department of Statistics, Faculty of Science, University of Colombo**, supervised by **[Dr. Chandima N.P.G. Arachchige](https://www.res.cmb.ac.lk/statistics/chandima-priyadarshani/)**.

Manual scoring of polysomnography (PSG) is the clinical gold standard for sleep staging, but it is time-consuming and prone to inter-rater variability. Convolutional networks perform well on this task, while Vision Transformers (ViTs) remain underexplored. Using a ViT means first turning one-dimensional signals into images, and it was unclear which signal-to-image transformation preserves sleep-stage information best. This study ranks the options before committing to a classifier.

## Abstract

Accurate sleep stage classification is essential for diagnosing sleep disorders, guiding treatment decisions, and advancing the understanding of brain function during sleep. Manual scoring of polysomnography (PSG) remains the clinical gold standard but is time-consuming and prone to inter-rater variability, motivating the need for reliable automated methods. While convolutional neural networks have demonstrated strong performance in this domain, Vision Transformers (ViTs) remain underexplored despite their ability to capture global dependencies. The primary objective of this study is to systematically evaluate the effectiveness of different signal-to-image (S2I) transformations for ViT-based sleep staging and to identify strategies that yield the most discriminative representations.

This study systematically evaluated 1,016 combinations of S2I methods, fusion strategies, and channel selections using the CAP Sleep Database, comprising 108 overnight PSG recordings annotated into standard 30-second sleep epochs. After preprocessing and artifact removal, four S2I transformations (spectrograms, GASF, GADF, and MTF) were applied, with fusion achieved through grid tiling or channel stacking. Embeddings extracted from a fine-tuned ViT-B/16 served as the basis for assessing stage separability using Fisher's Discriminant Ratio, Silhouette Coefficient, Davies-Bouldin Index, and Calinski-Harabasz Index. Composite scores derived from these metrics ranked strategies, and the top were validated through classification benchmarks with several base ViT models.

Results showed that GASF with grid tiling across all channels produced the most discriminative representations (composite score = 3.95), significantly outperforming other strategies (Mann-Whitney U, p < 0.001, r = 0.68, where p denotes significance level and r denotes rank-biserial correlation effect size). This top-ranked transformation was further assessed using five Vision Transformer architectures (ViT-B/32, DeiT-B, Swin-B, BEiT-B and PVT-Small) under five-fold stratified cross-validation. Performance was evaluated by overall accuracy and macro-averaged F1-score. ViT-B/32 achieved the highest results (83.1% accuracy, 79.4% macro-F1), with Swin-B and BEiT-B performing closely. These findings confirm that effective S2I strategies enhance ViT-based sleep staging and underscore the clinical potential of Vision Transformers.

**Keywords:** Sleep stage classification; Vision Transformers; Signal-to-image transformation; Deep learning

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ramanayakage2025separability,
  title     = {Representation Separability of Signal-to-Image Transformations for Sleep Stage Classification Using Pretrained Vision Transformers},
  author    = {Ramanayakage, Janith R.R.H. and Arachchige, Chandima N.P.G.},
  booktitle = {International Conference in Data Science (ICDS)},
  address   = {Colombo, Sri Lanka},
  year      = {2025}
}
```

## License

Released under the [MIT License](../../LICENSE).

## Contact

Janith R.R.H. Ramanayakage: janithramanayake523@gmail.com
