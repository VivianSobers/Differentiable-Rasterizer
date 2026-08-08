# Does the amortized model actually work?

The claim in the README is that a network can learn the *mapping* from image to
triangle scene, so that inference replaces hundreds of fitting iterations. This
is the record of checking that, which is worth reading before trusting any
PSNR number produced by `train.py`.

The short version: the first honest measurement showed the model **losing to a
flat colour fill**, and the cause was one line of architecture.

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

## The fix, and what it bought

`AdaptiveAvgPool2d(4)` keeps a 4x4 grid of features. Resolution-agnostic still,
coarse layout preserved. Same seed, same budget, same everything else:

| | `pool=1` | `pool=4` |
| --- | --- | --- |
| one-shot PSNR | 18.93 dB | **20.51 dB** |
| shuffled control | 18.10 dB | 17.10 dB |
| **input gain** | **0.83 dB** | **3.41 dB** |
| training loss | 0.01214 | **0.00810** |

**4.1x more of the score comes from reading the image.** Note that the shuffled
score went *down* while the one-shot score went up — exactly the signature of a
model becoming specific to its input rather than better at averaging. A model
that had merely gotten better in general would have lifted both.

## Does it actually save fitting iterations?

That is the claim the whole approach rests on, and it is separately testable:
run the same optimizer for the same budget from the prediction and from a
random start.

| 40 refinement steps, 48px | from prediction | from random |
| --- | --- | --- |
| `pool=1` | 24.44 dB | 23.49 dB |
| `pool=4` | **24.62 dB** | 23.49 dB |

A random start needs **16 steps** to reach what `pool=4` produces with none, up
from 12 for `pool=1` — a better model is worth more iterations skipped, which
is the metric behaving as intended.

Worth noting honestly: refinement washes most of the difference out. `pool=4`
starts 1.58 dB ahead one-shot and finishes only 0.18 dB ahead after 40 steps.
The head start is real but the fitter is good enough to mostly catch up, so
the value of a better model is in the iterations saved rather than in the final
quality reached.

## What is still wrong

Two of the three verdicts still fire at this budget, and neither is resolved:

- **It still loses to the mean-colour baseline**, though the gap fell from
  1.88 dB to 0.30 dB.
- **Mirror response is unchanged at 0.14.** The model responds to colour
  statistics far more than to layout — mirroring preserves colour exactly, and
  the prediction barely moves.

This run is small enough that "undertrained" is a live explanation for both:
48px, width 16, 8 epochs, 1024 images, a 3x3 feature map before pooling. The
honest statement is that `pool=4` is a clear improvement on an identical
budget, and that whether the remaining gap is budget or architecture is
**not yet measured**. The next lever to try, if a longer run does not close it,
is a head that reads spatial features per-triangle rather than a single flat
vector for all of them.

## Reproducing

```sh
python python/train.py --synthetic --epochs 8 --synthetic-size 1024 \
    --batch 16 --triangles 32 --size 48 --width 16 --pool 4 --out runs/pool4

python python/evaluate.py --checkpoint runs/pool4/best.pt --synthetic \
    --count 64 --refine-steps 40
```

Set `--pool 1` for the other arm. The evaluation is cheap; run it on anything
before believing its loss curve.
