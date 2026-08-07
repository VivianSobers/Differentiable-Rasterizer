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
faster — it is not, below roughly 64 triangles, where there are too few
gradient accumulators to spread contention across and a many-core CPU
parallelizing over batch items wins outright. `last_device()` reports what a
call actually used, which is worth checking: an earlier version of the binding
parsed `device` and silently ignored it, and timings alone did not reveal it.

Training is still staged so the GPUs are never idle: `precompute.py` fits a
corpus once in parallel across cores, and `train.py --pretrain` trains on those
pairs with no rasterizer in the loop at all.

The parameter-supervision term is deliberately decayed to zero over training.
Many different triangle sets render to the same image, so matching a particular
fit is a weaker signal than matching its output — useful as a warm start,
misleading as an objective.

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
| 128x128, 64 triangles | 0.52 ms | 0.10 ms | 5.2x |
| 256x256, 128 triangles | 3.65 ms | 0.20 ms | 18.2x |
| 512x512, 256 triangles | 25.8 ms | 0.70 ms | 36.8x |
| 512x512, 1024 triangles | 96.7 ms | 1.36 ms | **71x** |

| Forward + backward | CPU (26 cores) | RTX 4090 | Speedup |
| --- | --- | --- | --- |
| 128x128, 64 triangles | 2.81 ms | 2.18 ms | 1.3x |
| 256x256, 128 triangles | 22.2 ms | 16.9 ms | 1.3x |
| 256x256, 512 triangles | 85.8 ms | 18.3 ms | **4.7x** |

The speedup keeps growing with load and has not saturated at 1024 triangles,
which says the device is still being fed rather than waiting.

The most informative number is not a speedup at all. Going from 128 to 512
triangles at fixed resolution — 4x the geometry — costs the CPU **3.9x** more
time and the GPU **1.08x**:

| 256x256 | 128 triangles | 512 triangles | Cost of 4x geometry |
| --- | --- | --- | --- |
| CPU | 22.2 ms | 85.8 ms | 3.87x |
| GPU | 16.9 ms | 18.3 ms | **1.08x** |

The GPU absorbs four times the geometry almost for free, because at these sizes
the backward pass is bound by fixed overhead — buffer upload and readback — not
by arithmetic. That is the thing to attack next, and it is a very different
problem from the one the profile suggested on weaker hardware.

### Two backward implementations

Storing the canvas state per triangle (the obvious port of the CPU tape) reaches
384 MB at 512 triangles and 256px, and moving it costs more than it saves. So
there are two implementations, and `Recompute` is the default:

| Mode | Compute | Memory | 4090, 256px, 512 tris |
| --- | --- | --- | --- |
| `Taped` | O(T) | O(T x pixels) | 27.3 ms |
| `Recompute` | O(T²) | none | **18.3 ms** |

`Recompute` re-derives the canvas beneath each triangle rather than storing it,
and wins despite doing quadratically more arithmetic — the bounding-box cull
makes it far cheaper than its complexity suggests. Its margin *grows* with
triangle count (1.07x at 64 triangles, 1.49x at 512), and it held on both an
integrated AMD card and the 4090, which is about as much evidence as a design
choice like this can ask for. Both are kept, with a test asserting they agree.

### Batching, and where the GPU actually wins

`render_many` and `backward_many` submit a whole batch as one dispatch, which
removes the per-call overhead the profile pointed at. It works — **2.8-3.5x**
over dispatching one at a time, on both 4090 boxes.

It also does not do what I expected, which is the more useful result:

| batch of 64, 128px, 64 tris | CPU rayon (26 cores) | GPU one-by-one | GPU batched |
| --- | --- | --- | --- |
| worker-2 | **21.5 ms** | 135.0 ms | 45.4 ms |

Batching is 3x faster than not batching, and still **2x slower than the CPU**.
At small per-item work the CPU parallelizes across batch items nearly perfectly
— 26 cores on 64 independent images — while the GPU pays upload and readback for
a workload too small to amortize them. The GPU's 4.7x win at 256px with 512
triangles is real, and so is the CPU's 2x win at 128px with 64 triangles; they
are different regimes, not a contradiction.

The consequence for training: **routing the PyTorch layer to the GPU
unconditionally would be a regression** for typical configurations. `gpu_bench`
ends with a crossover table so the dispatch rule is read off measured data.

### The crossover, and what it exposed

> **These numbers predate the workgroup reduction below**, which made the GPU
> backward 2-7x faster. They are kept because they are what located the
> bottleneck; the `cpu` cells in particular should be re-measured before being
> used to choose a device. `./scripts/collect-gpu-report.sh` regenerates them.

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
triangles:

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

Speedup over the previous path, batch of 16:

| | 32 triangles | 128 triangles | 512 triangles |
| --- | --- | --- | --- |
| **64x64** | 2.70x | 2.54x | 2.07x |
| **128x128** | 4.47x | 3.24x | 2.83x |
| **256x256** | **6.87x** | 4.17x | 3.07x |

The *shape* of that table is the real result. Contention theory predicts the win
grows with pixel count (more threads queuing) and shrinks with triangle count
(more accumulator slots to spread across). Both hold, in both directions, across
all nine cells.

And the anomaly that started this is gone. Dispatch scaling per 4x pixels was
**12.6x**; it is now **~4.3x** — linear, which is what compute-bound looks like.

Barrier uniformity was the whole difficulty. `workgroupBarrier` is undefined
behaviour unless every thread reaches it, and the straightforward shader both
`return`s early for out-of-range pixels and `continue`s past culled triangles.
The restructure: an `in_range` predicate instead of the early return, triangles
processed in fixed-size chunks so barriers sit in control flow that depends only
on `n_tris`, and each chunk's partials flushed before the next clears them.
`workgroup_reduction_handles_non_multiple_sizes` covers the cases that would
expose a mistake — a 37x19 canvas with 33 triangles, where edge threads are
inactive and the last chunk is partial.

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
rewrite. On the 4090 it is 1.3-4.7x faster and binning is no longer the
bottleneck. The integrated result was real, but it generalized badly: a GPU
sharing system memory with the CPU is close to the worst case for this workload.
The conclusion changed because the measurement changed, which is the argument
for keeping `gpu_bench` in the repo rather than quoting numbers from one machine.

Cross-vendor agreement is itself a correctness signal: RADV (AMD) and NVIDIA's
driver both match the CPU forward pass to ~1e-6 and gradients to ~1e-4 relative,
having compiled the same WGSL through entirely different toolchains.

## Tests

```sh
cargo test --release          # 72 Rust tests
python -m unittest discover -s python -p "test_*.py"   # 43 Python tests
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

- **Per-workgroup gradient reduction.** The measured bottleneck: 86.6% of a
  batched backward call is dispatch, and the evidence points squarely at atomic
  contention. See the profile above for the design and its barrier-uniformity
  hazard.
- **Tiled binning.** Per-tile triangle lists would stop the shader visiting
  every (pixel, triangle) pair. Lower priority — the 4090 rejects culled pairs
  almost for free, and contention dominates.
- **Gaussian primitives** alongside triangles, closer to how 3D Gaussian
  splatting parameterizes scenes.
- **Perceptual loss** (LPIPS) in place of MSE, which over-rewards blur.
- **Adaptive triangle count** — split high-error triangles, prune invisible
  ones, rather than fixing the budget up front.

## License

MIT — see [LICENSE](LICENSE).
