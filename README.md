# Differentiable Rasterizer

A 2D triangle rasterizer that runs backwards. Give it a target image and it
gradient-descends the scene — vertex positions, colors, opacities — until the
render matches.

![convergence](docs/convergence.gif)

The trick that makes this possible is *soft* rasterization: a pixel's coverage
by a triangle is `sigmoid(signed_distance / sigma)` rather than a hard in/out
test. A hard test has zero gradient everywhere and an undefined one exactly at
the edge, so an optimizer never learns which way to move a vertex. Softening the
silhouette gives every pixel near an edge a real derivative.

| Target (176px) | 150 triangles, fitted, exported at 1024px |
| --- | --- |
| ![target](docs/target.png) | ![fit](docs/fit.png) |

![loss curve](docs/loss.png)

Loss falls **606x** in about 17 seconds on a single core. The fit runs at 176px
and the result exports at 1024px without refitting, because geometry is stored
in normalized coordinates rather than pixels.

## What's here

| Language | Share | Job |
| --- | --- | --- |
| **Rust** | ~70% | Rasterizer, analytic gradients, Adam, CLI, bindings |
| **Python** | ~18% | PyTorch layer, amortized model, training, charts |
| **TypeScript** | ~12% | Browser viewer — watch a fit converge live |

Each language is doing something the others would do worse. Rust owns the inner
loop, where per-pixel work over hundreds of iterations decides whether this is
interactive. Python owns training and presentation, where PyTorch and matplotlib
already exist. TypeScript owns the viewer, because a browser is the only place
someone can drop in their own photo and watch the optimizer work without
installing anything.

```
crates/diffrast/        Rust core
  src/raster.rs         Soft coverage, signed distance, forward render
  src/grad.rs           Reverse-mode gradients, parallel batch, FD reference
  src/optim.rs          Adam with per-parameter learning rates
  src/fit.rs            Fitting loop, sigma annealing, steppable Fitter
  src/canvas.rs         f32 RGB buffer, MSE loss, PNG + sRGB conversion
  src/scene.rs          Triangle/Scene types, flat parameter view
  src/serial.rs         Scene JSON export
  benches/raster.rs     Benchmark suite
crates/diffrast-gpu/    GPU rasterizer — WGSL compute shaders via wgpu
crates/diffrast-py/     PyO3 bindings — the rasterizer as a torch op
crates/diffrast-wasm/   WebAssembly bindings
python/diffrast/        Torch layer, model, datasets, plots
python/train.py         Distributed training
python/evaluate.py      Amortized model evaluation, with controls
python/sweep.py         Concurrent training sweeps, evaluated and tabulated
python/transfer.py      Generalization across resolution and scene complexity
web/                    TypeScript viewer
```

## Training through the renderer

The rasterizer is exposed to PyTorch as an ordinary differentiable op, so a
network can emit triangle parameters and be trained end-to-end against a
photometric loss:

```python
from diffrast.torch_layer import rasterize

params = model(image)                     # (B, T, 10)
render = rasterize(params, 128, 128)      # (B, 3, H, W)
loss = F.mse_loss(render, image)
loss.backward()                           # gradients reach the model weights
```

This is *amortized inverse graphics*: fitting one image takes hundreds of
gradient steps, but a trained network predicts a scene in a single forward pass,
which the fitter can then refine in far fewer iterations.

```sh
pip install torch                                    # then build the extension
cd crates/diffrast-py && maturin develop --release

python python/train.py --synthetic --epochs 20       # verifiable synthetic task
python python/precompute.py --data photos/ --out data/fits.pt
torchrun --nproc_per_node=2 python/train.py --pretrain data/fits.pt
```

**Choosing a backend.** The rasterizer runs on CPU or GPU, selected with
`--raster-device` (or `device=` on `rasterize`). This is independent of the
torch device: the extension carries its own wgpu context, and tensors always
cross the boundary as CPU float32.

`auto` picks from the measured crossover rather than assuming the GPU is
faster. On a discrete card it now is, at every size benchmarked; on an
integrated one it is not, at any size benchmarked, so the rule branches on the
adapter kind rather than pretending one threshold covers both.

`last_device()` reports what a call actually used, which is worth checking: an
earlier version of the binding parsed `device` and silently ignored it, and
timings alone did not reveal it. `policy_prefers_gpu()` answers the same
question ahead of time.

Training is still staged so the GPUs are never idle: `precompute.py` fits a
corpus once in parallel across cores, and `train.py --pretrain` trains on those
pairs with no rasterizer in the loop at all.

The parameter-supervision term is deliberately decayed to zero over training.
Many different triangle sets render to the same image, so matching a particular
fit is a weaker signal than matching its output — useful as a warm start,
misleading as an objective.

**A falling loss here is not evidence the model works**, which took a while to
appreciate. A network that ignored its input and emitted one generic scene
still drives the render loss down, because a mid-grey blob scores respectably
against almost any target. `evaluate.py` adds the controls that separate the
two — chiefly scoring each prediction against somebody else's target, and
comparing against a flat per-image colour fill:

```sh
python python/evaluate.py --checkpoint runs/amortized/best.pt --synthetic --refine-steps 40
```

The first honest measurement had the model **losing to the flat colour fill**.
Diagnosing why took two attempts. The first blamed `AdaptiveAvgPool2d(1)`
before the head — global average pooling is invariant to spatial permutation,
which for a task that is almost entirely about *where* triangles go discards
the signal the head needs. That is a real defect, and it is not what was doing
the damage: `--render-fraction` defaulted to 0.25, which with no parameter
supervision meant 75% of every batch contributed nothing to any loss. Fixing
the default moved the weaker architecture's input gain from 0.83 dB to 3.51 dB
on its own.

What the pooling actually governs is capacity. Asked to memorize 16 images,
`pool=4` reaches 27.07 dB against the 27.47 dB a direct fit achieves — it
matches the optimizer it is imitating — while `pool=1` stalls 2.3 dB short.

So capacity was never the constraint; data was. Trained on 160k synthetic
scenes the model reaches **8.29 dB of input gain**, beats the flat-colour
baseline by 4.52 dB, scores 0.90 on the mirror control, and its one-shot
prediction is worth **40 fitting iterations**. A five-configuration scaling
sweep puts numbers on which lever matters: 4x the data is worth +1.57 dB of
margin, 2x the triangles +0.60, and 2x the model width +0.24.

That sweep also found that the model does **not** transfer across resolution —
5.6 dB lost to a 1.33x change — which contradicted a design note in `model.py`
about adaptive pooling. [docs/AMORTIZED.md](docs/AMORTIZED.md) has the full
table, the two different ways of grading it that were both wrong, and the
correction.

## Benchmarks

Measured on 16 cores with `cargo bench`:

| Operation | Time |
| --- | --- |
| Render 256x256, 64 triangles | 3.4 ms |
| Forward + backward 128x128, 64 triangles | 2.9 ms |
| Forward + backward 256x256, 256 triangles | 45.4 ms |
| `backward_batch` of 64 (128px, 64 tris) | 25.8 ms — **0.40 ms/item** |

Batching gives a **7.1x speedup** per item (2.85 ms alone vs 0.40 ms in a batch
of 64). The single fit loop is sequential by nature — each triangle attenuates
the gradient the next one reads — so the parallelism lives one level up, over
independent images. That is exactly the shape training needs.

The patch-based tape uses **15x less memory** than storing a canvas per
triangle: 13 MB versus 197 MB for 256 triangles at 256px.

## The GPU path

`crates/diffrast-gpu` runs the same renderer and the same gradients as WGSL
compute shaders. Correctness is defined as agreement with the CPU, and checked:
forward matches to **1e-6**, gradients to **2e-6** relative.

```sh
cargo run --release --bin gpu_bench
```

The port hinges on one inversion. The CPU loops over triangles, and each one
composites onto what the last left behind — a dependency that serializes the
whole loop. The GPU gives each thread one *pixel* and has it walk the triangle
list itself. The sequential chain still exists; it just runs inside a thread
instead of across them.

Measured on an **RTX 4090** against a 26-core i9-13900. (Reproduce with
`./scripts/collect-gpu-report.sh`; `docs/gpu-report.txt` holds the raw run.)

| Forward | CPU (26 cores) | RTX 4090 | Speedup |
| --- | --- | --- | --- |
| 128x128, 64 triangles | 0.52 ms | 0.11 ms | 4.7x |
| 256x256, 128 triangles | 3.53 ms | 0.26 ms | 13.6x |
| 512x512, 256 triangles | 24.5 ms | 0.66 ms | 37.0x |
| 512x512, 1024 triangles | 96.5 ms | 1.38 ms | **70x** |

| Forward + backward | CPU (26 cores) | RTX 4090 | Speedup |
| --- | --- | --- | --- |
| 128x128, 64 triangles | 2.70 ms | 2.22 ms | 1.2x |
| 256x256, 128 triangles | 21.1 ms | 17.3 ms | 1.2x |
| 256x256, 512 triangles | 88.5 ms | 18.8 ms | **4.7x** |

The speedup keeps growing with load and has not saturated at 1024 triangles,
which says the device is still being fed rather than waiting.

The most informative number is not a speedup at all. Going from 128 to 512
triangles at fixed resolution — 4x the geometry — costs the CPU **4.2x** more
time and the GPU **1.09x**:

| 256x256 | 128 triangles | 512 triangles | Cost of 4x geometry |
| --- | --- | --- | --- |
| CPU | 21.1 ms | 88.5 ms | 4.19x |
| GPU | 17.3 ms | 18.8 ms | **1.09x** |

The GPU absorbs four times the geometry almost for free, because these are
single unbatched calls and fixed per-call overhead swamps the arithmetic. That
observation is what the rest of this section is about: the overhead turned out
to be three separate things, only one of which was the one it looked like.

### Two backward implementations

Storing the canvas state per triangle (the obvious port of the CPU tape) reaches
384 MB at 512 triangles and 256px, and moving it costs more than it saves. So
there are two implementations, and `Recompute` is the default:

| Mode | Compute | Memory | 4090, 256px, 512 tris |
| --- | --- | --- | --- |
| `Taped` | O(T) | O(T x pixels) | 27.5 ms |
| `Recompute` | O(T²) | none | **18.8 ms** |

`Recompute` re-derives the canvas beneath each triangle rather than storing it,
and wins despite doing quadratically more arithmetic — the bounding-box cull
makes it far cheaper than its complexity suggests. Its margin *grows* with
triangle count (1.07x at 64 triangles, 1.49x at 512), and it held on both an
integrated AMD card and the 4090, which is about as much evidence as a design
choice like this can ask for. Both are kept, with a test asserting they agree.

### Batching, and where the GPU actually wins

`render_many` and `backward_many` submit a whole batch as one dispatch, which
removes the per-call overhead the profile pointed at. It works — **6-8x** over
dispatching one at a time:

| batch of 64, 128px, 64 tris | CPU rayon (26 cores) | GPU one-by-one | GPU batched |
| --- | --- | --- | --- |
| worker-2, at the time | 21.5 ms | 135.0 ms | **45.4 ms** |

Batching was 3x better than not batching and still **2x worse than the CPU**.
At small per-item work 26 cores parallelize across batch items nearly
perfectly, while the GPU paid upload and readback for a workload too small to
amortize them.

That measurement is why the PyTorch layer was never routed to the GPU
unconditionally, and it was the right call on the evidence available. It is
also no longer true — the same row now reads 5.0 ms, and
[the final crossover](#the-crossover-finally) has the GPU ahead everywhere on
that card. `gpu_bench` ends with a crossover table precisely so the dispatch
rule tracks the hardware instead of a conclusion that has now aged out twice.

### The crossover, and what it exposed

> These are the **pre-reduction** numbers. They are kept because they are the
> diagnostic — the shape of this table is what located the bottleneck. The
> current crossover, which is what you should actually dispatch on, is
> [further down](#the-crossover-after-the-fix).

Batch of 16 on an idle 4090 vs 26 CPU cores — `gpu/cpu` above 1.0 means the GPU
wins:

| | 32 triangles | 128 triangles | 512 triangles |
| --- | --- | --- | --- |
| **64x64** | 0.42x | **1.17x** | **1.95x** |
| **128x128** | 0.34x | **1.17x** | **1.91x** |
| **256x256** | 0.12x | 0.49x | **1.17x** |

More triangles help the GPU, as expected. More *pixels* hurt it, which is
backwards — more pixels is more parallelism. Pulling on that:

| GPU cost for 4x the work | via pixels | via triangles |
| --- | --- | --- |
| 32 triangles | **10.5x** | — |
| 128 -> 512 triangles | — | 1.9x |

Scaling superlinearly in pixels and sublinearly in triangles is not how compute
behaves; it is how *transfers* behave. The cause was in this repo, not the
hardware: `backward_many` rendered by calling `render_many`, which copied the
batch to the host — and then immediately uploaded it straight back as an input
to the backward pass. 12.6 MB round-tripped per call at 256px, for data that
never needed to leave the device.

The rendered batch now stays in GPU memory between the two dispatches. Images
are still read back once, because the loss is reduced on the host; moving that
reduction onto the GPU would remove the last per-pixel transfer, and is the next
thing worth doing.

That fix was real but small — 3-9% at 256px. It was not the bottleneck, and
predicting that it would be was wrong. Which is what the phase breakdown is for.

### Where the time actually goes

`backward_many_timed` reports each phase. On an idle 4090, batch of 16, 32
triangles — again, this is the **pre-reduction** profile, and the
[post-fix one](#what-is-left-now-that-it-is-not-the-atomics) looks nothing like
it:

| resolution | pack | alloc | dispatch | readback | loss | total |
| --- | --- | --- | --- | --- | --- | --- |
| 64x64 | 0.02 ms | 0.26 ms | 0.76 ms | 0.15 ms | 0.06 ms | 1.25 ms |
| 128x128 | 0.02 ms | 0.99 ms | 5.84 ms | 0.41 ms | 0.18 ms | 7.44 ms |
| 256x256 | 0.03 ms | 6.09 ms | **73.43 ms** | 4.43 ms | 0.79 ms | 84.77 ms |

Dispatch is **86.6%**. Transfers and host-side work together are under 7%, which
settles it: the cost is GPU execution, and every plumbing optimization —
including the one above — was addressing the wrong 7%.

### The bottleneck is atomic contention

Gradients accumulate per triangle while threads run per pixel, so every pixel
that a triangle covers does a compare-exchange against the same ten floats.
Four independent observations all point there:

1. **Dispatch scales 12.6x for 4x the pixels.** More pixels means more threads
   contending for the same accumulators, and CAS retries grow faster than
   linearly with contention.
2. **Per-triangle cost falls 7.2x as triangles increase.** At 256px, going from
   32 to 512 triangles is 16x the work but only 2.2x the time — because 320
   accumulator slots become 5120, and the contention spreads out.
3. **The forward pass, which has no atomics, scales cleanly** and reaches 73.5x.
   It shares the same geometry, the same culling, the same memory layout. The
   accumulation is the only difference.
4. **The GPU wins exactly where contention is lowest** — many triangles, which
   is also where the crossover table flips to `GPU`.

### The fix, and what it confirmed

`ReduceMode::Workgroup` accumulates into workgroup-shared memory first, then
performs one global atomic per triangle per *workgroup* instead of one per
pixel — up to 64x fewer global atomics at a 64-thread workgroup. It is the
default.

Speedup over the previous path, batch of 16, on an idle 4090:

| | 32 triangles | 128 triangles | 512 triangles |
| --- | --- | --- | --- |
| **64x64** | 1.96x | 1.69x | 1.21x |
| **128x128** | 4.49x | 3.41x | 2.04x |
| **256x256** | **21.5x** | 10.4x | 3.96x |

(Measured after the overhead work below. The 256px figures were 6.59x, 5.19x
and 3.31x when both paths still carried ~11 ms of fixed allocation and
transfer; removing a constant from both sides of a ratio inflates it, and the
later numbers flatter the reduction for a reason that has nothing to do with
the reduction.)

The *shape* of that table is the real result. Contention theory predicts the win
grows with pixel count (more threads queuing) and shrinks with triangle count
(more accumulator slots to spread across).

Both axes hold monotonically in all nine cells: every column rises with
resolution, every row falls with triangle count. That is as clean a
confirmation as a prediction of this kind gets — and it got *cleaner* once the
overhead work below stopped fixed transfer costs from masking the trend on the
smaller canvases.

And the anomaly that started this is gone. Dispatch scaling per 4x pixels was
**12.6x**; it is now **2.4-3.1x** — linear or better, which is what
compute-bound looks like.

Barrier uniformity was the whole difficulty. `workgroupBarrier` is undefined
behaviour unless every thread reaches it, and the straightforward shader both
`return`s early for out-of-range pixels and `continue`s past culled triangles.
The restructure: an `in_range` predicate instead of the early return, triangles
processed in fixed-size chunks so barriers sit in control flow that depends only
on `n_tris`, and each chunk's partials flushed before the next clears them.
`workgroup_reduction_handles_non_multiple_sizes` covers the cases that would
expose a mistake — a 37x19 canvas with 33 triangles, where edge threads are
inactive and the last chunk is partial.

### The crossover, after the fix

Same measurement as [before](#the-crossover-and-what-it-exposed) — batch of 16,
idle 4090 vs 26 CPU cores, `gpu/cpu` above 1.0 means the GPU wins:

| | 32 triangles | 128 triangles | 512 triangles |
| --- | --- | --- | --- |
| **64x64** | 0.83x | **2.04x** | **2.37x** |
| **128x128** | **1.02x** | **3.61x** | **3.52x** |
| **256x256** | 0.79x | **2.41x** | **3.38x** |

The pixel penalty is gone, which is the clearest confirmation that the diagnosis
was right. At 512 triangles the GPU's margin used to *shrink* as resolution grew
— 1.95x, 1.91x, 1.17x — and now holds flat at 2.37x, 3.52x, 3.38x. Resolution
stopped being the thing that hurt.

What survives is a real CPU-wins column at 32 triangles. That one is not
contention — see [what was left](#what-is-left-now-that-it-is-not-the-atomics),
which is where it goes.

### What is left, now that it is not the atomics

The same phase breakdown, re-run — batch of 16, 32 triangles:

| resolution | pack | alloc | dispatch | readback | loss | total |
| --- | --- | --- | --- | --- | --- | --- |
| 64x64 | 0.02 ms | 0.26 ms | 0.26 ms | 0.11 ms | 0.12 ms | 0.77 ms |
| 128x128 | 0.02 ms | 0.99 ms | 0.63 ms | 0.36 ms | 0.22 ms | 2.22 ms |
| 256x256 | 0.03 ms | **5.85 ms** | 1.95 ms | **4.34 ms** | 0.87 ms | 13.04 ms |

Dispatch at 256px went from 73.43 ms to 1.95 ms — a **38x** drop — and the
profile inverted with it. Dispatch was 86.6% of the call and is now 15%. Buffer
allocation and readback, the "under 7%" that I explicitly dismissed as the wrong
thing to optimize, are now **78%**.

Both statements were true when measured. Removing a dominant cost promotes
whatever was sitting behind it, so a profile is only ever a statement about the
current bottleneck — not a ranking of what matters.

It also explains the 32-triangle column exactly. At 256px the GPU's actual
compute is 1.95 ms against the CPU's 11.90 ms — a 6x win it never gets to bank,
because 11 ms of allocation and transfer sit on top of it. The CPU is not
winning that cell on arithmetic; it is winning it on not having a bus.

### Removing the overhead

Two changes, both aimed at that 78%.

**Buffer pooling.** Every call allocated a fresh image buffer, target buffer,
gradient accumulator and staging buffer, then dropped them — at identical sizes,
thousands of times, because that is what a training loop is. They now come from
a pool keyed on `(usage, size)`. The gradient accumulator is the one that cannot
simply be reused as-is, since it is added into rather than overwritten; it gets
a device-side `clear_buffer` instead of an upload of zeros. The concatenated
host-side copy of the target batch went with it — targets are written straight
into the buffer per image, which removes a 12.6 MB `memcpy` per call.

**Device-side loss.** `loss_batch.wgsl` reduces per-image MSE with one workgroup
per batch item, so the rendered batch never leaves the device. Previously the
host read back 12.6 MB in order to produce 16 floats. Gradients and losses are
then fetched in a single mapped readback rather than two.

On the 4090, batch of 16 at 256px:

| phase | before | after | |
| --- | --- | --- | --- |
| pack | 0.03 ms | 0.02 ms | |
| alloc | 5.85 ms | 1.19 ms | 4.9x |
| dispatch | 1.95 ms | 1.96 ms | unchanged, as intended |
| readback | 4.34 ms | 0.03 ms | **145x** |
| loss | 0.87 ms | 0.00 ms | now on the device |
| **total** | **13.04 ms** | **3.20 ms** | **4.1x** |

Everything that is not dispatch fell from 11.06 ms to 1.24 ms — **8.9x**. The
call is now 61% dispatch, which is what it should have been all along.

I predicted 3x here and got 8.9x, so the prediction was right in direction and
wrong in size — the two effects compound, because pooling also removes the
staging buffer that the readback would have allocated.

The same change is worth only 12% on the integrated card, where dispatch is 40
of 46 ms and dominates regardless. Overhead work pays off in proportion to how
little of the time is real work.

Reuse is asserted rather than assumed: `buffers_are_reused_across_identical_calls`
checks that a repeated call allocates *nothing*. A pool that silently never hit
would be invisible in a timing, and `pool_stats()` exists so the test can look
instead of infer. Across a full `gpu_bench` run it reports 1745 hits to 31
misses.

That test also found something I had asserted without checking. It originally
required the two calls to return *bit-identical* gradients, which passed on an
integrated card and failed on the 4090 — by one ULP, on one value in sixty.
**The GPU backward pass is not bit-reproducible**, and cannot be: each
workgroup adds its partial into a triangle's accumulator through a global
atomic, the scheduler picks the order, and float addition is not associative. A
4090's 128 SMs reorder enough to show it where 12 compute units did not. The
test now uses a tolerance four orders of magnitude tighter than the defect it
guards — a buffer reused without clearing comes back doubled, not different in
the last bit.

### The crossover, finally

Batch of 16, idle 4090 vs 26 CPU cores, after all three changes:

| | 32 triangles | 128 triangles | 512 triangles |
| --- | --- | --- | --- |
| **64x64** | **1.85x** | **1.70x** | **2.26x** |
| **128x128** | **4.06x** | **4.96x** | **3.89x** |
| **256x256** | **3.64x** | **6.03x** | **3.89x** |

Every cell, including the 32-triangle column that the CPU won twice. At 256x256
with 32 triangles the GPU has gone 0.12x → 0.79x → **3.64x** across the three
rounds of this investigation, without the shader's arithmetic changing at all.

The batched backward pass moved with it — same work, one dispatch instead of N:

| batch of 64, 128px, 64 tris | CPU rayon | GPU one-by-one | GPU batched |
| --- | --- | --- | --- |
| before | 21.5 ms | 135.0 ms | 45.4 ms |
| now | 20.8 ms | 179.3 ms | **5.0 ms** |

9x on the batched path, and the gap over dispatching one at a time went 3x →
**35.8x**, because per-call overhead is exactly what was removed.

**This is one machine.** The same benchmark on an integrated AMD part has the
CPU winning all nine cells by 2-3x, and that is not a constant to fold into one
threshold — an integrated GPU is competing for the memory bandwidth of the
cores it is racing. `device="auto"` therefore branches on whether the adapter
is discrete, with both measured tables pinned in
`test_policy_matches_the_discrete_crossover` and its integrated twin.

### The general lesson

The profile said "the GPU is slow at high resolution". The first answer was "it
copies every pixel across PCIe twice" — true, and worth 3-9%. The real answer
was "thousands of threads are queuing for ten floats". Two wrong diagnoses
preceded the right one, and what separated them was instrumenting the thing
instead of reasoning about it.

### How batching is implemented

The scene index rides in the dispatch's `z` dimension and triangles are addressed
as `batch * n_tris + i`, so every accessor in `common.wgsl` works unchanged.
Scenes in a batch must share a triangle count, which training satisfies by
construction and which is checked rather than assumed — a ragged batch would
silently read a neighbouring scene's triangles instead of failing.

Only the recompute strategy is offered batched. A batched tape would need
`batch x triangles x pixels x 3` floats — 12 GB for a batch of 32 at 256px with
512 triangles, which exceeds a 4090's memory for a workload that otherwise fits
comfortably.

### Two 4090s, two different answers

The same benchmark on two nominally identical boxes:

| forward, 512px, 1024 tris | GPU time | speedup |
| --- | --- | --- |
| worker-2 (idle card) | 1.35 ms | **71x** |
| worker-1 (desktop daemon on card) | 3.27 ms | 28x |

CPU times were within 5% of each other, so the difference is the GPU: worker-1
runs `gnome-remote-desktop-daemon` holding ~1 GB of its card, and a newer
driver. Benchmarks belong on the idle box. This is why the report script now
lists GPU processes before it starts — the earlier version would have reported
the 28x without any indication of why.

### A note on how this was measured

An earlier revision of this file reported, from an integrated AMD Radeon 860M,
that the GPU backward pass was *slower* than the CPU and needed a tiled-binning
rewrite. On the 4090 it is 1.2-4.7x faster and binning is no longer the
bottleneck. The integrated result was real, but it generalized badly: a GPU
sharing system memory with the CPU is close to the worst case for this workload.
The conclusion changed because the measurement changed, which is the argument
for keeping `gpu_bench` in the repo rather than quoting numbers from one machine.

Cross-vendor agreement is itself a correctness signal: RADV (AMD) and NVIDIA's
driver both match the CPU forward pass to ~1e-6 and gradients to ~1e-4 relative,
having compiled the same WGSL through entirely different toolchains.

## Tests

```sh
cargo test --release          # 86 Rust tests
python -m unittest discover -s python -p "test_*.py"   # 61 Python tests
cargo clippy --all-targets    # clean
cargo fmt --all -- --check
```

The tests worth knowing about:

- `analytic_gradient_matches_finite_differences` — every parameter against a
  numerical gradient, in Rust.
- `test_gradient_matches_numerical_differentiation` — the same check again from
  the PyTorch side, so the binding layer can't quietly corrupt the gradient in
  transit. Agreement is within **0.34%**, cosine similarity 0.9992.
- `gradient_points_downhill` — a step along the negative gradient really does
  reduce the loss.
- `stepping_matches_the_batch_loop` — pins the browser's incremental path to the
  CLI's batch path so they cannot drift apart.
- `batch_gradients_match_sequential_ones` — parallelism changes the schedule,
  never the numbers.

**On validating a gradient numerically.** Finite-difference error here is
U-shaped in the step size, and picking a step off that curve's floor produces a
convincing-looking failure that is entirely an artifact:

| step | relative error |
| --- | --- |
| 1e-4 | 1.49% |
| 5e-4 | **0.34%** |
| 1e-3 | 0.54% |
| 5e-3 | 10.2% |
| 1e-2 | 20.0% |

Below the floor, float32 cancellation dominates — the two loss evaluations are
so close that their difference is mostly rounding noise. Above it, the step is
large enough that the measurement captures the function's curvature rather than
its slope. The tests use 5e-4.

A related limitation: the check runs at `sigma = 0.03`, much softer than a
finished fit. At production sharpness the sigmoid saturates and central
differences measure noise at any step size. That constrains the *check*, not the
gradient, but it does mean the analytic path can't be validated directly at the
sigma a fit ends on.

## Roadmap

- **Tiled binning.** Per-tile triangle lists would stop the shader visiting
  every (pixel, triangle) pair. Now the *only* remaining lever: dispatch is
  61% of a backward call, up from 15%, purely because everything around it got
  cheaper. The 4090 still rejects culled pairs almost for free, so the win
  would come from the `O(T²)` recompute rather than from the culling.
- **Gaussian primitives** alongside triangles, closer to how 3D Gaussian
  splatting parameterizes scenes.
- **Perceptual loss** (LPIPS) in place of MSE, which over-rewards blur.
- **Adaptive triangle count** — split high-error triangles, prune invisible
  ones, rather than fixing the budget up front.

## License

MIT — see [LICENSE](LICENSE).
