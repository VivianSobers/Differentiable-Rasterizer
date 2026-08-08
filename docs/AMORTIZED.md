# Does the amortized model actually work?

The claim in the README is that a network can learn the *mapping* from image to
triangle scene, so that inference replaces hundreds of fitting iterations. This
is the record of checking that, which is worth reading before trusting any
PSNR number produced by `train.py`.

The short version: the first honest measurement showed the model **losing to a
flat colour fill**. Diagnosing why took two attempts — the first blamed an
architectural choice that turned out to be mostly a confounded default — and
the current answer is that the model has ample capacity and does not
generalize.

## Why the training loss was not evidence

`train.py` renders the predicted scene and scores it against the target. That
number went down. It would also go down for a model that ignored its input
entirely and emitted a single generic scene, because the images in a set share
a great deal — a mid-grey blob scores respectably against almost anything.

Loss curves cannot separate "learned the mapping" from "learned the average".
So `evaluate.py` reports three controls alongside the headline number.

**Shuffled control.** Score each predicted render against a *different*
target. Whatever score survives that is what the model earns without looking at
its input. The gap — `input_gain_db` — is the part attributable to actually
reading the image, and it is the number that matters.

**Mean-colour baseline.** Fill each image with its own average colour. It is
the cheapest prediction that still adapts to the input, it takes no training,
and a model below it is worse than useless however good its PSNR looks alone.

**Mirror response.** A horizontally mirrored image is a different scene: every
triangle has to move. A model reading spatial layout should respond about as
strongly to mirroring as to an unrelated image, giving a ratio near 1.0. Since
mirroring leaves colour statistics untouched, this ratio separates *where* the
model is sensitive to layout from where it is only sensitive to colour.

## The first measurement

A small model (32 triangles, width 16, 48px, 8 epochs over 1024 synthetic
scenes):

```
one-shot PSNR            18.93 dB
  vs shuffled targets    18.10 dB
  input gain              0.83 dB
mean-colour baseline     20.81 dB
mirror response           0.15

verdict: model barely uses its input; loses to a flat colour fill;
         largely spatially blind
```

0.83 dB of input gain against a 20.81 dB baseline it never reaches. The model
had learned an expensive constant with slight colour adaptation.

## The cause

```python
self.pool = nn.AdaptiveAvgPool2d(1)
```

Global average pooling is **invariant to spatial permutation**. It reports how
much of each feature is present somewhere in the image and discards where. For
classification that is a feature — object identity should not depend on
position. For inverse graphics it is close to the worst possible bottleneck,
because position is nearly the entire task: the head was asked to place 32
triangles from a vector that had been stripped of all layout information.

The comment justifying it was about resolution-agnosticism — train at 64px,
infer at 256px without reshaping the head. That benefit is real. It just does
not require pooling all the way to 1x1.

`test_pooling_to_one_discards_spatial_layout` asserts the invariance directly
on the pooling layer, so this is a property of the operation rather than a
story about one training run.

## The fix

`AdaptiveAvgPool2d(4)` keeps a 4x4 grid of features. Resolution-agnostic still,
coarse layout preserved.

## A confounded comparison, and the correction

The first A/B looked spectacular — input gain 0.83 dB for `pool=1` against
3.41 dB for `pool=4`, a 4.1x improvement. It was wrong, and the way it was
wrong is worth more than the result would have been.

`train.py` defaulted to `--render-fraction 0.25`: render a quarter of each
batch, on the reasoning that rendering dominates a step and a noisier gradient
is a fine trade. That reasoning holds *only when a second loss term covers the
rest of the batch*. With `--synthetic` there is no parameter supervision, so
the render loss is the only loss and 75% of every batch contributed nothing —
loaded, moved to the device, discarded.

Both arms were handicapped identically, so the comparison was fair. It was
also run at a quarter of the intended data, and that turned out to be where
most of the difference lived. Re-run with the full batch:

| held-out, 1024 scenes, 8 epochs | `pool=1` | `pool=4` |
| --- | --- | --- |
| one-shot PSNR | 20.55 dB | 20.64 dB |
| **input gain** | 3.51 dB | 3.62 dB |
| mirror response | 0.12 | 0.20 |
| training loss | 0.00805 | **0.00764** |

`pool=1`'s input gain went from 0.83 dB to 3.51 dB purely by not throwing the
batch away. The architecture was not what was crippling it; data starvation
was, and the pooling choice merely made starvation hurt more.

The default is now `1.0` unless `--pretrain` supplies the second loss term,
and an explicit fractional render without one prints a warning. A default that
silently discards most of a batch is a bad default, and it cost a real
experiment here before anyone noticed.

## Where the architecture does matter

Capacity, which the generalization numbers above cannot see. Train each arm to
memorize 16 images — a task they should be able to solve outright, since the
targets *are* 32-triangle renders — and compare against what a direct fit
achieves on the same images:

| 16 images, 800 steps, full batch | PSNR | vs ceiling |
| --- | --- | --- |
| direct fit, 200 Adam steps | **27.47 dB** | — |
| `pool=4` | 27.07 dB | -0.40 dB |
| `pool=1` | 25.21 dB | -2.26 dB |
| per-image mean colour | 20.59 dB | — |

`pool=4` essentially saturates the ceiling: it memorizes as well as the fitter
optimizes. `pool=1` cannot, by 2.3 dB, and that gap survives the full batch.
So the pooling choice is real — it just governs how well the model *can* fit,
not how well it currently generalizes.

The ceiling itself is the other useful number here. 27.5 dB, not 40: recovering
a triangle scene from its render is a hard non-convex problem even with the
right triangle count and unlimited iterations. Every model number on this page
should be read against 27.5, not against perfection.

## Does it actually save fitting iterations?

That is the claim the whole approach rests on, and it is separately testable:
run the same optimizer for the same budget from the prediction and from a
random start.

| 40 refinement steps, 48px | from prediction | from random |
| --- | --- | --- |
| `pool=1` | 24.33 dB | 23.49 dB |
| `pool=4` | **24.84 dB** | 23.49 dB |

A random start needs **16 steps** to reach what either model produces with
none. That is the amortization claim holding, if modestly: one forward pass is
worth about 16 fitting iterations here.

Worth noting honestly: refinement compresses the difference between the two
arms. They start 0.09 dB apart one-shot and finish 0.51 dB apart after 40
steps, both far below the 27.5 dB a longer direct fit reaches. The head start
is real, and it is small — the value of a better model is in iterations saved,
not in the final quality reached.

## What is still wrong

Both arms still fail two of the three verdicts on held-out data:

- **They lose to the mean-colour baseline** — 20.64 dB against 20.81 dB. Close,
  but a model that cost 700k parameters should beat a flat fill outright.
- **Mirror response is 0.20.** The prediction responds to colour statistics far
  more than to layout. Mirroring an image leaves its colours untouched and
  demands that every triangle move; the model barely reacts.

The capacity result above says this is **not** a representational limit:
`pool=4` reaches 27.07 dB when it only has to memorize, within 0.4 dB of what
direct optimization achieves. It can produce scenes this good. It just cannot
yet produce the *right* one for an image it has not seen.

That is a generalization gap, and 1024 scenes over 8 epochs at 48px is a
plausible cause on its own. The experiment that settles it is more data at
higher resolution, which is what the run in [RUNBOOK.md](RUNBOOK.md) does.

If a 40k-scene run does not close it, the next lever is the head: it currently
reads one flat pooled vector and emits all triangles from it, so every triangle
sees the same global summary. A head that lets each triangle attend to its own
region — which is what the mirror-response number is really complaining about —
is the structural change that would follow.

## Reproducing

```sh
# generalization arm
python python/train.py --synthetic --epochs 8 --synthetic-size 1024 \
    --batch 16 --triangles 32 --size 48 --width 16 --pool 4 --out runs/pool4

# capacity arm: same model, 16 images, long enough to memorize them
python python/train.py --synthetic --epochs 400 --synthetic-size 16 \
    --batch 8 --triangles 32 --size 48 --width 16 --pool 4 --out runs/overfit4

python python/evaluate.py --checkpoint runs/pool4/best.pt --synthetic \
    --count 64 --refine-steps 40
```

Set `--pool 1` for the other arm of either. Both need `--render-fraction 1.0`
on any build before the default changed, or the comparison measures data
starvation instead of architecture.

The evaluation is cheap; run it on anything before believing its loss curve.
