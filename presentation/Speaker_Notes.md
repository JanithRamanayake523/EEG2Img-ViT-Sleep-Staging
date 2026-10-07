# Speaker Notes (10-minute talk + Q&A)

**Pacing:** about 130 words per minute. Timings are cumulative; the talk ends at 9:45, leaving a 15-second buffer. Pause on each **bold takeaway**. Do not read slides aloud; say the point, then let the number sit.

**One-sentence story to keep in mind:** *Turn each EEG epoch into an image the right way (GASF, 128 Hz DWT), fine-tune a pre-trained ViT, and you get 86.2% accuracy, 10 points above a CNN.*

---

## Slide 1: Title [0:00-0:25]

Good morning, everyone. I'm Janith Ramanayake, and this is joint work with Dr. Chandima Arachchige at the University of Colombo. Our question is simple: when you feed sleep signals to a Vision Transformer, **how should you turn the signal into an image?** Most people just pick a spectrogram. We tested that choice systematically.

---

## Slide 2: Why automate sleep staging? [0:25-1:15]

Sleep disorders such as insomnia, apnoea and narcolepsy affect tens of millions of people. Diagnosis relies on polysomnography, where an expert manually scores every 30-second epoch of a whole night.

Three numbers frame the problem. Scoring one recording takes **two to four hours**. Even experienced experts agree only **82.6%** of the time. So the labels themselves are noisy, and that is the ceiling we are working against.

Deep learning has moved from CNNs and RNNs to Vision Transformers. ViTs are powerful, but they expect images. That raises our question: how do we make images out of signals?

*Transition:* So what has been done, and what is missing?

---

## Slide 3: Research gap and objectives [1:15-2:15]

Prior ViT work for sleep staging uses spectrograms or raw signals. Four things are missing.

- **Nobody has compared** GASF, GADF and MTF for ViTs. These are signal-to-image encodings from time-series imaging.
- Nobody has asked which **sampling rate** is best *before* imaging.
- Most models are trained from scratch, so the value of **pre-training** is unclear.

That gives us four experiments, one per question: image transformation; sampling rate with FFT versus DWT; pre-trained versus from scratch, balanced versus imbalanced; and finally seven architectures, including a ResNet-50 baseline. Each experiment feeds its winner into the next one.

*Transition:* First, the data.

---

## Slide 4: Dataset [2:15-3:00]

We used the **CAP Sleep Database** from PhysioNet: 80 recordings, 6 healthy subjects and 74 patients. That gives **80,667 thirty-second epochs**, 16 channels at 512 Hz, labelled with six R&K stages: Wake, S1 to S4, and REM.

The classes are imbalanced. S2 makes up almost half, and S1 only 6.5%. I'll come back to that.

Most important for credibility: we use **subject-wise 5-fold cross-validation**. No subject ever appears in both train and test, so the numbers are not inflated by leakage.

---

## Slide 5: Proposed pipeline [3:00-4:00]

*Point at the figure, left to right.*

This is the whole system in five stages.

1. **Raw PSG:** EEG derivations plus EOG, EMG and ECG, 16 channels at 512 Hz.
2. **Preprocessing:** band-pass 0.3 to 45 Hz, a 50 Hz notch, ICA for artifact removal, then downsampling. The star marks the result of Experiment 2: DWT at 128 Hz.
3. **Image transformation:** the four candidates, GASF, GADF, MTF and spectrogram. GASF is highlighted because it wins.
4. **Multi-channel composition:** each channel becomes one small image, and the 16 tile into a **4 by 4 grid**, which we resize to 224 by 224 for the ViT.
5. **Classification:** an **ImageNet pre-trained ViT-B/16** predicts one of six stages.

The orange and gold highlights are the winning choices, so you can read the best pipeline straight off this figure.

---

## Slide 6: Four signal-to-image transformations [4:00-5:00]

Each signal is first normalised to minus one to one, so every value can be treated as an angle.

- **GASF** takes the cosine of the *sum* of angles. It preserves global temporal correlation.
- **GADF** takes the sine of the *difference*. It is less globally coherent.
- **MTF** quantises into 32 bins and encodes Markov transition probabilities. It loses information in quantisation, and it is slow: **5 seconds per epoch**, against 0.2 for GASF.
- The **spectrogram** is a standard STFT with a 256-sample Hamming window. Its time-frequency resolution is fixed.

All four go through the identical 4 by 4 composition into ViT-B/16, so differences come from the encoding alone.

---

## Slide 7: Result 1, GASF performs best [5:00-5:50]

*Experiment 1.* **GASF reaches 78.5% accuracy**, macro-F1 0.773. Next come GADF at 77.2%, MTF at 75.8% and the spectrogram at 74.1%.

An honest reading: the GASF-GADF gap is about the size of the fold-to-fold standard deviation, so treat them as **roughly equivalent**. MTF and the spectrogram are clearly behind. Since GASF is also 25 times faster than MTF, it is the practical choice.

*Takeaway:* the angular-summation encodings beat the standard spectrogram.

---

## Slide 8: Result 2, DWT at 128 Hz is optimal [5:50-6:40]

*Experiment 2.* Taking GASF forward, we downsampled the 512 Hz signals to 256, 128 and 64 Hz, using either FFT resampling or DWT.

FFT gets steadily worse as the rate drops. DWT is different: it **peaks at 128 Hz with 80.3%**, macro-F1 0.791, which is **1.8 points above the original 512 Hz** signal. My interpretation is that wavelet downsampling acts as a denoiser. Below 128 Hz, information loss takes over.

One caution: the margin over DWT at 256 Hz is small, so I would not over-claim a sharp optimum. The robust finding is that **DWT beats FFT at every reduced rate**, and that lower resolution does not have to hurt.

---

## Slide 9: Result 3, Pre-training adds 11+ points [6:40-7:30]

*Experiment 3.* This is the biggest effect in the paper. With GASF and 128 Hz DWT, the pre-trained ViT-B/16 reaches **86.2%** on balanced data, against **74.8%** from scratch: **plus 11.4 points**. On imbalanced data the gain is **plus 12.2**, 80.5% against 68.3%.

It is surprising that ImageNet features help on GASF images, which look nothing like photographs. The ViT has learned generic patterns of texture and structure that transfer.

Balancing, using DCGAN synthetic samples generated **only inside training folds**, adds about 5.7 points. The test folds stay real, so there is no leakage.

---

## Slide 10: Result 4, ViT-B/16 beats other models [7:30-8:20]

*Experiment 4.* We ran seven architectures on the best pipeline. **ViT-B/16 is best at 86.2%**, macro-F1 0.853. Swin-B follows at 85.1%, then BEiT, DeiT and ViT-B/32.

Two things to note. First, the **ResNet-50 CNN baseline reaches 76.2%**, a full **10 points lower**, so the transformer advantage is real. Second, the lead over Swin-B is only 1.1 points, so several transformers are close. If compute matters, **PVT-Small gets 81.8% with only 24.5 million parameters**. That is the efficient option.

---

## Slide 11: Class-wise results and limitations [8:20-9:05]

Per class, S2 and Wake are strongest. **S1 is weakest**: it is the rarest stage at 6.5% of epochs and the most ambiguous even for human scorers. It also benefits most from balancing.

I want to be upfront about limitations:

- a **single database**, with no cross-dataset test;
- **R&K labels**, not the current AASM scoring;
- **no formal significance tests**;
- the 4 by 4 **channel ordering was not ablated**.

Read the results as strong evidence on one dataset, not a final verdict.

---

## Slide 12: Conclusions and future work [9:05-9:45]

To sum up, the effective pipeline has three parts: **GASF encoding, DWT downsampling to 128 Hz, and a fine-tuned ImageNet ViT-B/16**. It reaches **86.2% accuracy and 0.853 macro-F1**, 10 points above a ResNet-50.

Next, we plan multimodal fusion, lightweight ViTs for edge deployment, and cross-dataset validation.

---

## Slide 13: Thank you [9:45-10:00]

Thank you. The paper and slides are in the repository, and I'm happy to take questions.

---

# Q&A preparation

**Why GASF over a spectrogram?**
GASF keeps the full temporal ordering and pairwise relations at the full image resolution. A spectrogram trades time resolution for frequency resolution. Our result (78.5% against 74.1%) supports this, though we did not test why.

**Is GASF really better than GADF?**
Not clearly. The 1.3-point gap is about the fold standard deviation, so we call them comparable. We chose GASF because it is nominally best and the formulation is simple.

**Why does DWT at 128 Hz beat the original signal?**
Our hypothesis is denoising by wavelet decomposition, and the smaller image is easier for the ViT to use. We have not isolated the cause, so we state it as a hypothesis.

**Doesn't DCGAN balancing risk leakage?**
No. Synthetic samples are generated only from training folds. Test folds are all real data.

**Why does ImageNet pre-training help on non-natural images?**
Low-level filters such as edges and textures transfer across domains. The gain was 11 to 12 points on both balanced and imbalanced data, so the effect is consistent.

**How does this compare with other published results?**
Be careful: datasets, label schemes and splits differ, so direct comparison is not fair. Our claim is about the relative ranking of design choices under one fixed protocol.

**Why a 4 by 4 grid, and does channel order matter?**
16 channels fit a square grid, which suits ViT input. We did not ablate the ordering. This is listed as a limitation.

**Is it deployable?**
ViT-B/16 is heavy. PVT-Small gives 81.8% with 24.5M parameters, and lightweight ViTs are our next step.

**Does it generalise to AASM or other datasets?**
Unknown. We used R&K labels on a single database, and cross-dataset validation is future work.

---

# Delivery tips

- **Checkpoints:** be on Slide 5 by 3:00 and Slide 9 by 6:40. If you are behind, shorten Slide 6 and the Slide 11 limitations to one sentence each.
- **If you are ahead of time:** add a sentence on why the labels' 82.6% agreement caps achievable accuracy.
- Keep the four result slides (7 to 10) crisp. The audience should leave with three numbers: **78.5%** (GASF), **+11.4 points** (pre-training), **86.2%** (best model).
- Slow down on the Slide 5 figure. It is the one slide where the audience must see the whole system.
