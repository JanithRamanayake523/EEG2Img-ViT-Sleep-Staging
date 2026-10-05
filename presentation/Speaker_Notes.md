# Speaker Notes: Paper ID 45 (10-minute talk)

## Slide 1: Systematic Evaluation of Signal-to-Image Transformation Pipelines for Vision Transformer-based Sleep Stage Classification

[0:00-0:30] Good morning. I'm presenting our systematic evaluation of how to turn raw sleep signals into images for Vision Transformers.

## Slide 2: Why automate sleep staging?

[0:30-1:30] Sleep disorders are common. The gold standard is manual scoring of polysomnography, which takes hours per night and even experts agree only about 82.6% of the time. Deep learning has moved from CNNs and RNNs to Vision Transformers, which need image inputs.

## Slide 3: Research gap and objectives

[1:30-2:30] Existing ViT sleep studies use spectrograms or raw signals. Nobody has compared GASF, GADF and MTF for ViTs, nobody has tuned the sampling rate before imaging, and most models are trained from scratch. So we designed four experiments.

## Slide 4: Dataset: CAP Sleep Database

[2:30-3:30] We used 80 recordings from the CAP Sleep Database on PhysioNet, 16 channels at 512 Hz, 80,667 thirty-second epochs, six R&K stages. Wake, S1, S2, S3, S4, REM. S2 dominates at almost half. Evaluation is subject-wise 5-fold cross-validation so no subject appears in both train and test.

## Slide 5: Proposed pipeline

[3:30-4:30] Five stages. Raw PSG, preprocessing with filters and ICA artifact removal then FFT or DWT downsampling, image transformation, composing the 16 channel images into a 4 by 4 grid of 224 by 224, and classification with an ImageNet pre-trained ViT-B/16.

## Slide 6: Four signal-to-image transformations

[4:30-5:30] Each normalised signal in [-1,1]. GASF and GADF map values to angles and encode summed or differenced angular relations. MTF quantises into 32 bins and encodes Markov transitions. The spectrogram uses a 256-sample Hamming STFT. Sixteen channel images are tiled into a 4 by 4 composite.

## Slide 7: Result 1: GASF performs best

[5:30-6:30] GASF gets 78.5% accuracy, GADF 77.2, MTF 75.8, spectrogram 74.1. The GASF-GADF gap is comparable to fold standard deviation so treat them as similar; MTF and spectrogram are clearly behind. MTF is also 25 times slower per epoch.

## Slide 8: Result 2: DWT at 128 Hz is optimal

[6:30-7:15] We downsampled the original 512 Hz signals with FFT or DWT. FFT gets monotonically worse. DWT peaks at 128 Hz with 80.3%, 1.8 points above the original signal, probably denoising. Below 128 Hz information loss dominates. The margin over 256 Hz DWT is small, so interpret with caution.

## Slide 9: Result 3: Pre-training adds 11+ points

[7:15-8:00] With GASF plus DWT at 128 Hz, the ImageNet pre-trained ViT-B/16 reaches 86.2% on balanced data versus 74.8% from scratch, an 11.4 point gain, and 12.2 points on imbalanced data. Class balance came from DCGAN synthetic augmentation applied only to training folds.

## Slide 10: Result 4: ViT-B/16 beats other models

[8:00-8:45] Seven architectures on the best pipeline. ViT-B/16 is best at 86.2%, Swin-B 85.1, BEiT, DeiT, ViT-B/32 follow. PVT-Small gets 81.8 with only 24.5M parameters. The ResNet-50 CNN baseline reaches 76.2, 10 points lower. The ViT-B/16 lead over Swin-B is only 1.1 points.

## Slide 11: Class-wise results and limitations

[8:45-9:30] Per-class F1 under balanced training: S2 and Wake are strongest; S1, the rarest stage at 6.5% of epochs, is weakest and benefits most from balancing. Limitations: single database, R&K not AASM scoring, no formal significance tests, the 4 by 4 channel ordering was not ablated, and ViT-B/16 is computationally heavy.

## Slide 12: Conclusions and future work

[9:30-10:00] The effective pipeline is GASF encoding, DWT to 128 Hz, and a fine-tuned ImageNet ViT-B/16, reaching 86.2% accuracy and 0.853 macro-F1, 10 points above ResNet-50. Future work: multimodal fusion, lightweight ViTs for edge deployment, cross-dataset validation. Thank you, happy to take questions.
