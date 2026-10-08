# Speaker Notes: full spoken script (10 minutes + Q&A)

**How to use this.** Everything in normal text is what you say, word for word. Words in *[italics and brackets]* are actions: a pause, a gesture, a click. About 1,300 words at a relaxed 130 words per minute. Timings are cumulative. Do not memorise it; read it aloud twice and then say it in your own words. The meaning matters, not the exact sentences.

**If you only remember one thing:** *Turn each EEG epoch into an image the right way (GASF, 128 Hz DWT), fine-tune a pre-trained ViT, and you get 86.2% accuracy, 10 points above a CNN.*

---

## Slide 1: Title [0:00-0:25]

*[Smile, look at the audience, wait a second before you start.]*

Good morning, everyone, and thank you for being here. I'm Janith Ramanayake from the University of Colombo, and this is joint work with my co-author, Chandima Arachchige.

*[Slight pause.]*

I want to start with a simple question. If you want a Vision Transformer to read sleep signals, **how should you turn the signal into an image?** Most people pick a spectrogram and move on. We decided to test that choice properly.

*[Click.]*

---

## Slide 2: Why automate sleep staging? [0:25-1:15]

Let's start with why this matters. Sleep disorders like insomnia, apnoea and narcolepsy are very common. In the US alone, somewhere between 50 and 70 million adults live with one.

To diagnose them, an expert has to score a whole night of sleep recording by hand, one 30-second epoch at a time. *[Point to the first number.]* That takes **one to three hours** for a single recording. And it is not even consistent. *[Point to the second number.]* Experienced experts agree with each other only **82.6%** of the time.

So the task is slow, and it is also subjective. And there is one more complication. *[Point to the scoring card.]* There are two scoring standards in use. The older one, R&K, has six stages: Wake, S1 to S4, and REM. The newer AASM standard has five, because it merges S3 and S4 into a single stage, N3. We use R&K here, because that is how this database is labelled. That is exactly why people want to automate it.

Deep learning has moved from CNNs and RNNs to Vision Transformers. They are powerful, but they expect images as input. And that is where our question comes from: how do we get images out of signals?

*[Click.]*

---

## Slide 3: Research gap and objectives [1:10-2:00]

So what has been done already? *[Gesture to the left column.]*

Earlier work with Vision Transformers for sleep staging mostly used spectrograms, or raw signals. And that leaves three open questions.

First, there are other ways to turn a signal into an image, called GASF, GADF and MTF. Nobody has compared them for Vision Transformers. Second, nobody has asked what sampling rate works best *before* you make the image. And third, it is unclear whether pre-training on ImageNet actually helps on these images.

*[Gesture to the right column.]*

So we designed four experiments, one for each question, plus a fourth to compare architectures. First, the image transformation. Second, the sampling rate. Third, pre-trained against from-scratch. And fourth, seven different models, including a ResNet-50 as a CNN baseline. Each experiment passes its winner to the next one.

*[Click.]*

---

## Slide 4: Dataset [2:00-2:40]

Now, the data. We used the CAP Sleep Database from PhysioNet. It has **80 recordings**: 6 healthy people and 74 patients. In total that is about **80,000 thirty-second epochs**, from 16 channels recorded at 512 Hz, and each epoch is labelled with one of six stages: Wake, S1 to S4, and REM.

The classes are not balanced. Stage 2 makes up almost half of all epochs, while S1 is only 6.5%. Keep that in mind, because I'll come back to it.

*[Slow down here.]* One thing I want you to remember. We evaluate with **subject-wise five-fold cross-validation**. So a person is either in training or in testing, never both. That means our numbers are not inflated by the model recognising the same person twice.

*[Click.]*

---

## Slide 5: Proposed pipeline [2:40-3:35]

*[Slow down. Point at the figure and move your hand left to right as you talk.]*

This figure is the whole system, so let me walk you through it from left to right.

Stage one is the raw data: EEG, plus eye, muscle and heart signals, 16 channels at 512 Hz.

Stage two is cleaning. We band-pass filter, remove the 50 Hz mains noise, and remove artifacts with ICA. Then we downsample. The star here marks the best choice we found: wavelet downsampling to 128 Hz.

Stage three is where each signal becomes an image. We tried four methods, and the highlighted one, GASF, is the winner.

In stage four, each channel gives one small image, and we tile all 16 into a **four-by-four grid**. That becomes one picture per epoch.

And in stage five, that picture goes into a Vision Transformer, ViT-B/16, pre-trained on ImageNet, which predicts one of six sleep stages.

So if you want to know our best pipeline, you can read it straight off this figure.

*[Click.]*

---

## Slide 6: Four signal-to-image transformations [3:35-4:15]

Let me explain the four transformations we compared.

*[Point at each card as you name it.]*

The first two work with angles. The **Gramian Angular Summation Field**, GASF, sums angles, and it keeps the global time relationships in the signal. The **Gramian Angular Difference Field**, GADF, uses the difference of angles, and it is a bit less coherent overall.

The **Markov Transition Field**, MTF, groups the signal into 32 levels and records how often it moves from one level to another. It throws away some detail in that grouping, and it is slow: five seconds per epoch.

And the spectrogram is the standard choice: a short-time Fourier transform. Its time and frequency resolution is fixed.

All four go through exactly the same tiling and the same ViT, so any difference in results comes from the transformation alone.

*[Click.]*

---

## Slide 7: Same epoch, four images [4:15-4:50]

Before I show you numbers, let's look at what these actually produce. This is one 30-second epoch from one subject, at the original 512 Hz.

*[Point left to right.]*

GADF and GASF share the same block structure, because they are built from the same angles, but GASF shows more fine detail. The spectrogram is mostly dark. Most of its energy sits in a few bands. And MTF looks busy and blocky, which comes from that grouping into 32 levels.

So the same signal gives very different pictures. Now the question is: which picture does a Vision Transformer understand best?

*[Click.]*

---

## Slide 8: Result 1, GASF performs best [4:50-5:30]

*[Let the number appear, then speak.]*

Here is the answer. **GASF reaches 78.5% accuracy.** Then GADF at 77.2%, MTF at 75.8%, and the spectrogram at 74.1%.

I should be honest about what this shows. The gap between GASF and GADF is about the same size as the variation between folds, so I would call those two comparable. But MTF and the spectrogram are clearly behind.

And there is a practical point too. *[Point at the 5.0 s card.]* MTF takes five seconds per epoch. GASF takes 0.2. So GASF is the best choice, and also the cheapest one.

*[Click.]*

---

## Slide 9: Downsampling: FFT vs DWT [5:30-6:05]

Next question: do we really need all 512 Hz? Smaller signals mean smaller images and faster training.

Here is the same epoch as a GASF image, at 64, 128 and 256 Hz, next to the original. *[Point to the top row.]* The top row uses FFT resampling. These images barely change. *[Point to the bottom row.]* The bottom row uses wavelet downsampling, DWT, and here the image changes clearly at the lower rates. At 64 Hz it is much smoother.

So the two methods give the model different inputs. Which one is better?

*[Click.]*

---

## Slide 10: Result 2, DWT at 128 Hz is optimal [6:05-6:50]

*[Point at the purple line.]*

The purple line is DWT, the grey line is FFT. FFT gets steadily worse as we reduce the rate. DWT does something surprising. It **peaks at 128 Hz with 80.3%**, which is **1.8 points better than the original signal at 512 Hz**.

Why would less data be better? My guess is that the wavelet step removes some noise. But I want to be clear that this is an interpretation, not something we proved. Below 128 Hz, we start losing real information.

Also, the gap between 128 and 256 Hz is small, so I would not claim a sharp optimum. The solid finding is this: **DWT beats FFT at every reduced rate**, and lowering the resolution does not have to hurt.

*[Click.]*

---

## Slide 11: Class balancing with DCGAN [6:50-7:25]

Before the next result, one more ingredient. Remember that S1 was only 6.5% of the data. To fix the imbalance, we used a DCGAN to generate synthetic GASF images.

*[Point left, then right.]*

On the left is a real image. On the right is one generated by the DCGAN for the same class. It is noisier, but it keeps the same four-by-four block layout and the same overall texture.

And one rule matters a lot. Synthetic images go **only into the training folds**. The test folds stay completely real, so there is no leakage.

*[Click.]*

---

## Slide 12: Result 3, Pre-training adds 11+ points [7:25-8:05]

This is the biggest effect in the whole paper.

On balanced data, the pre-trained ViT reaches **86.2%**. The same model trained from scratch gets **74.8%**. That is a gain of **11.4 points** just from starting with pre-training. On imbalanced data the gain is even bigger, **12.2 points**.

I find that interesting, because GASF images look nothing like photographs. Still, what the network learned from ImageNet about edges and textures seems to carry over.

And balancing the classes with the DCGAN adds about another 5.7 points.

*[Click.]*

---

## Slide 13: Result 4, ViT-B/16 beats other models [8:05-8:50]

Finally, which model is best? We compared seven on the best pipeline.

**ViT-B/16 comes first at 86.2%.** Swin-B is close behind at 85.1%. The CNN baseline, ResNet-50, reaches 76.2%. That is **10 points lower**, so the advantage of transformers here is real.

Two honest notes. First, the lead over Swin-B is only 1.1 points, so a few transformers are close. Second, ViT-B/16 is heavy. If you care about compute, PVT-Small gets 81.8% with only 24.5 million parameters. That is the efficient option.

*[Click.]*

---

## Slide 14: Class-wise results and limitations [8:50-9:30]

Looking at individual stages, S2 and Wake are classified best. **S1 is the weakest.** It is the rarest stage, and it is also the one that benefits most from balancing.

Now, the limitations, and I want to be upfront about them. *[Count on your fingers.]* One, we used a single database, so we have no cross-dataset test. Two, the labels follow the older R&K rules, not the current AASM rules. Three, we did not run formal significance tests. And four, we did not test whether the order of the channels in the grid matters.

So please read our results as strong evidence on one dataset, not as the final word.

*[Click.]*

---

## Slide 15: Conclusions and future work [9:30-9:50]

*[Slow down. This is the take-home message.]*

To sum up. The best pipeline has three parts: **GASF images, DWT downsampling to 128 Hz, and a fine-tuned, pre-trained ViT-B/16.** Together they reach **86.2% accuracy**, which is 10 points above a ResNet-50.

Next, we want to combine more signal types, build lighter models that could run on small devices, and test on other datasets.

*[Click.]*

---

## Slide 16: Thank you [9:50-10:00]

*[Smile, look up.]*

Thank you very much. If you want the code or the paper, you can scan this QR code. I'd be happy to take any questions.

---

# Q&A: how to answer (say these in your own words)

**Why GASF over a spectrogram?**
"GASF keeps the order of the signal and the relationships between every pair of points, at full image resolution. A spectrogram has to trade time detail for frequency detail. The results support that, 78.5% against 74.1%, but we did not test the reason directly."

**Is GASF really better than GADF?**
"Not clearly. The gap is about the size of the fold-to-fold variation, so I call them comparable. We went with GASF because it scored best and the formula is simple."

**Why does DWT at 128 Hz beat the original signal?**
"My hypothesis is that wavelet downsampling removes noise. But we did not isolate the cause, so I present it as a hypothesis."

**Doesn't the DCGAN risk leakage?**
"No. Synthetic images are generated and used only in the training folds. The test folds are all real data."

**Why does ImageNet pre-training help on these images?**
"Low-level features like edges and textures seem to transfer across domains. The gain was 11 to 12 points both on balanced and imbalanced data, so the effect is consistent."

**How does this compare with other published work?**
"Direct comparison is hard, because datasets, label schemes and splits differ. Our claim is about how design choices rank against each other under one fixed protocol."

**Does the order of channels in the 4 by 4 grid matter?**
"We did not test it. It is a limitation, and a good thing to ablate next."

**Could this run on a device?**
"ViT-B/16 is heavy. PVT-Small gives 81.8% with 24.5 million parameters, and lighter ViTs are our next step."

**Does it work on AASM labels or other datasets?**
"We do not know yet. We used R&K labels on one database. Cross-dataset validation is future work."

**If you do not know the answer:**
"That is a good question. We have not tested it, so I do not want to guess. It is something we would like to look at."

---

# Delivery tips

- **Checkpoints:** be on Slide 6 by about 3:35 and Slide 12 by about 7:25. If you run behind, say Slides 7, 9 and 11 in one sentence each.
- **Pauses:** stop for one second after each big number: 78.5%, 80.3%, 86.2%, 11.4 points. Silence makes numbers land.
- **Eye contact:** look at the audience for the first sentence of each slide, then glance at the slide when you point to something.
- **Speed:** most people speed up when nervous. If you notice, breathe and slow down on Slides 5 and 15.
- **Three numbers to leave them with:** **78.5%** (GASF), **+11.4 points** (pre-training), **86.2%** (best model).
