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

## What scaling actually buys

A five-configuration sweep on an RTX 4090, 5.4 hours, every model graded on one
held-out benchmark (64 triangles at 96px, 256 images):

| run | tris | train px | width | scenes | margin | gain | refined | steps saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data-40k | 64 | 96 | 32 | 40k | 2.95 | 6.41 | - | - |
| **data-160k** | 64 | 96 | 32 | 160k | **4.52** | **8.29** | 30.12 | **40** |
| tris-128 | 128 | 96 | 32 | 40k | 3.55 | 7.64 | **34.85** | 18 |
| width-64 | 64 | 96 | 64 | 40k | 3.19 | 6.85 | 29.28 | 26 |
| res-128 | 64 | 128 | 32 | 40k | -0.57 | 2.69 | 28.45 | 8 |
| big | 128 | 128 | 64 | 160k | -1.17 | 2.51 | 33.87 | 5 |

Reading it needs care, and the first two attempts at this table were both
misleading — see the section after this one.

**Data dominates.** Against the 40k baseline, one variable at a time: 4x the
scenes is worth **+1.57 dB** of margin, 2x the triangles +0.60, and 2x the
width **+0.24**. Width buys almost nothing, which the capacity experiment
already predicted — the model could reach the fitter's own quality when
memorizing, so more parameters were never what it was short of.

One caveat that the sweep cannot resolve: these runs held epochs fixed, so 4x
the scenes was also 4x the gradient steps. "More data" and "more compute" are
not yet separated. The `diversity` plan exists to do exactly that, holding
samples and steps identical while varying only how often a scene repeats.

**One-shot quality and refinement quality are different objectives**, and no
single model wins both. `data-160k` produces the best *starting point* — its
one-shot prediction is worth 40 fitting iterations. `tris-128` produces the
best *destination*, 4.7 dB better after 100 steps, while saving less than half
as many. Triangle count drives where refinement ends up; data drives how good
the prediction is on its own. If the goal is replacing fitting iterations,
`data-160k` is the better model despite losing on refined PSNR.

## Diversity or compute?

The scaling sweep could not tell those apart: it held epochs fixed, so 4x the
scenes was also 4x the gradient steps. The `diversity` plan separates them.
Every arm sees **exactly 9.6M samples in exactly 150,000 steps** — identical
compute, identical optimizer schedule — and varies only how often a scene
repeats.

| unique scenes | epochs | margin | gain | one-shot | refined | steps saved |
| --- | --- | --- | --- | --- | --- | --- |
| 40k | 240 | 3.48 | 7.27 | 22.16 | 30.24 | 29 |
| 160k | 60 | 4.43 | 8.18 | 23.11 | 30.21 | 39 |
| 640k | 15 | **4.56** | **8.29** | **23.23** | 30.30 | **41** |

**Diversity is real and it saturates.** Going 40k -> 160k buys **+0.95 dB** of
margin at fixed compute; going 160k -> 640k buys **+0.13 dB**. The second
quadrupling is worth a seventh of the first.

Comparing across the two sweeps decomposes the original confounded result. The
same 40k scenes trained 4x longer (60 -> 240 epochs) gained +0.53 dB; 4x the
scenes at fixed compute gained +0.95 dB. Both mattered, diversity roughly twice
as much — and the earlier "+1.57 dB from 4x data" was about 60% diversity and
40% compute.

The practical conclusion is that **both levers are now spent**. 160k scenes is
the knee, more compute is worth half of what more data was, and neither is far
from flat. Whatever comes next has to be architecture or objective, not scale.

## Refinement erases the difference between models

The most surprising number in that table is the one that does not move.

After 100 refinement steps the three models land at **30.24, 30.21 and 30.30
dB** — a spread of 0.08 dB, from models whose one-shot predictions differ by
1.07 dB. The better model does not reach a better place. It arrives sooner:
29, 39 and 41 fitting iterations skipped.

That is worth being precise about, because it constrains what amortization is
*for* here. A better predictor is not a route to better final quality; the
fitter reaches the same optimum regardless of where it starts. It is purely a
way to spend fewer iterations getting there.

The refinement traces also show every model losing ground on its first step —
23.23 -> 22.21 before recovering by step 3. Adam at `lr=0.02` is tuned for a
random initialization, and a good initialization is exactly what it damages.
Two of the ~40 saved iterations are spent undoing that, and nothing has tried
lowering the rate when starting from a prediction.

## Does resolution invariance have to be trained?

Not answered yet, and the first attempt was invalid — worth recording because
the failure was in the harness rather than the idea.

| trained at | margin | gain | one-shot |
| --- | --- | --- | --- |
| fixed 96px | 1.96 | 5.60 | 20.64 |
| jitter 64/96/128 | **-0.37** | 2.88 | 18.31 |
| jitter 48-160 | 1.98 | 5.56 | 20.66 |

Wider jitter matching no jitter while *narrower* jitter collapses is not a
curve, it is noise — and the cause is `best.pt`. Model selection compared each
epoch's training loss against the others', which assumes they are on the same
scale. Under `--sizes` they are not: every epoch trains at a different
resolution and MSE is not comparable across them, so the saved checkpoint was
chosen by whichever resolution happened to score lowest.

Selection now runs on a held-out set at one fixed resolution. The effect is
visible immediately on a four-epoch smoke test: the final epoch had the lowest
*training* loss and a worse held-out loss than the epoch before it, so the two
criteria disagree about which checkpoint to keep.

The experiment needs re-running before it says anything. What can already be
said is that a model trained at one resolution does not transfer, and that
none of these three managed to beat a flat colour fill by more than 2 dB —
against 4.56 dB for the best fixed-resolution model on more data.

### The re-run

Selection now runs on a held-out set at one fixed resolution (`42a86e6`). Along
the way, re-running `--sizes` with workers also surfaced a second, unrelated
bug: `--sizes` needs a resolution change to reach already-forked DataLoader
workers, and the only mechanism the old code had for that was
`persistent_workers=False`, re-forking every epoch. By the time that happens
the main process has already used the GPU rasterizer — every training step
does — and forking after that deadlocks the child on a lock the parent's
now-absent thread was holding. `--workers 0` never hit it; `--workers 4` hung
on the very first batch, reproduced twice on a smoke test, and is fixed
(`1e9f2ce`) by forking workers exactly once, before anything touches the GPU,
and passing per-epoch resolution changes into the already-running workers
through a `multiprocessing.Value` instead of by re-forking.

Three arms, 60 epochs, 40k scenes, otherwise identical (width 32, pool 4, 64
triangles): fixed 96px, jitter 64/96/128px, jitter 48-160px. All graded first
on the shared 96px benchmark, `sweep.py`'s standard table:

| trained at | margin@96 | gain@96 | one-shot@96 |
| --- | --- | --- | --- |
| fixed 96px | **3.22** | **6.82** | **21.90** |
| jitter 64/96/128 | 3.06 | 6.65 | 21.74 |
| jitter 48-160 | 2.96 | 6.45 | 21.64 |

On shared home turf, jitter costs a little — 0.16 to 0.26 dB of margin versus
training at 96px alone, roughly in proportion to how wide the jitter range is.
That is the expected price of spreading one fixed epoch budget across more
resolutions instead of concentrating it at one.

The real question — does jitter training actually transfer better? — needs
evaluating each checkpoint away from 96px too, which is not what `sweep.py`
does automatically: its "native" column only re-evaluates when a run's base
`--size` differs from the benchmark's, and here it's 96 for all three configs
(`--sizes` varies the *training* resolution, not the `--size` flag), so
margin equals native trivially and says nothing about transfer. Evaluating all
three checkpoints at 64px and 128px directly:

| margin (dB over flat colour) | 64px | 96px | 128px |
| --- | --- | --- | --- |
| fixed 96px | -1.94 | **3.22** | -0.44 |
| jitter 64/96/128 | **-0.22** | 3.06 | **-0.09** |
| jitter 48-160 | -1.01 | 2.96 | -0.20 |

Jitter training helps, measurably. At 64px, jitter 64/96/128 closes 1.72 dB of
the gap over fixed-96px (-1.94 -> -0.22); jitter 48-160 closes 0.93 dB. At
128px the gap is smaller to start and jitter still shrinks it (-0.44 -> -0.09
and -0.20 respectively).

**It does not close it.** Every arm, including the widest jitter tested, still
loses to a flat colour fill at both 64px and 128px. Adaptive pooling makes
varying resolution *possible*; training across resolutions makes it *less
bad*; nothing tried here makes it *free*. "Trained at one resolution, runs at
another" is a claim about tensor shapes standing in for a much weaker one
about accuracy, at every setting measured so far — jittered training included.

One more thing worth recording plainly: narrower jitter (64/96/128) beat wider
jitter (48-160) at *both* off-resolution points, despite the wide arm's range
literally containing the eval points too. The likely explanation is training
density, not range — three sizes concentrate more of a fixed epoch budget at
exactly 64/96/128 than five sizes spread it across, and the wide arm's 48px and
160px scenes, which the eval never visits, dilute it further. Widening the
jitter range is not free either.

This also puts the earlier invalid table in context: the corrected
`jitter 64/96/128` run scores 3.06 dB margin at 96px, nothing like the invalid
run's -0.37 — a swing attributable entirely to the checkpoint-selection bug the
invalid attempt was measuring through, not to anything about resolution jitter.

## Refinement rate: hypothesis rejected

Every refinement trace loses ground on its first step — 23.23 -> 22.21 before
recovering by step 3 — which looked like Adam at `lr=0.02` being tuned for a
random start and damaging a good one. Swept on the best model, 100 steps:

| lr | from prediction | from random |
| --- | --- | --- |
| 0.05 | 28.12 | 25.19 |
| **0.02** | **30.31** | 25.58 |
| 0.01 | 29.99 | 24.75 |
| 0.005 | 29.42 | 23.56 |

The default was already the best of the four. The first-step dip is real and
costs about two iterations, and every rate that avoids it costs more than that
in convergence. Nothing to fix.

Worth noting how the "steps saved" metric misleads here: it *rises* to 89 at
`lr=0.005`, which looks like a better head start and is nothing of the kind —
the random baseline slowed down too. That number is only comparable at a fixed
learning rate.

## Two ways of grading that were both wrong

The first version of this table scored each model on a held-out set matching
its *own* triangle count and resolution. Every model therefore sat a different
exam: absolute PSNR reflected how hard that model's eval set happened to be,
the mean-colour baseline moved with it, and the columns could not be compared.

Fixing that — one shared benchmark for everything — introduced the opposite
problem, and produced the two negative rows above. `res-128` and `big` are the
only models trained at 128px, and they are the only models scored below the
baseline. They did not fail. They were graded out of distribution:

| `big`, identical weights | margin |
| --- | --- |
| graded at 128px, its training resolution | **+4.44 dB** |
| graded at 96px, the shared benchmark | **-1.17 dB** |

**5.6 dB lost to a 1.33x change in resolution.** `TriangleNet` pools adaptively
so that a model trained at one size accepts any other, and the comment
justifying that said the model could be "trained at 64px and run at 256px". That
is true about tensor shapes and false about accuracy, which is a distinction the
architecture's design note did not make and now does.

There is no single fair exam once models are trained at different resolutions.
The sweep now reports both — `margin` on the shared benchmark and `native` at
each model's own resolution — because `margin << native` is the signature of a
model that is fine and does not transfer, and one column cannot show it.
Whether that transfer gap is a property of the architecture or merely of
training at a single resolution is what `--plan resolution` and `--sizes` are
for.

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
