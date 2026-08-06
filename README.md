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

**A note on where the time goes.** The rasterizer runs on CPU, so a naive
end-to-end loop is bottlenecked by rendering and the GPUs idle. Training is
staged around that: `precompute.py` fits a corpus once, in parallel across
cores, and `train.py --pretrain` then trains on those pairs as pure GPU work.
End-to-end fine-tuning applies the render loss to a fraction of each batch
(`--render-fraction`), which makes steps far cheaper at the cost of a noisier
gradient. A GPU rasterizer would remove the constraint entirely — it's the
biggest single improvement left, and it's on the roadmap below.

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

## Running it

**Command line.** Fit an image and write every artifact:

```sh
cargo run --release --bin fit -- photo.jpg --tris 200 --iters 2000 --save-every 10
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tris` | 128 | triangles in the scene |
| `--iters` | 1500 | optimizer steps |
| `--size` | 192 | longest side of the fit |
| `--out` | `out` | output directory |
| `--save-every` | 0 (off) | write a frame every N iterations |
| `--export` | 1024 | longest side of the final render |
| `--seed` | 0 | RNG seed |
| `--patience` | 250 | stop after N stalled iterations (0 disables) |

Omit the image to fit a generated synthetic target. Outputs are `fit.png`,
`target.png`, `scene.json`, and `loss.csv`.

**Python** — the same fit plus charts and an animated GIF:

```sh
pip install -r python/requirements.txt
python python/report.py photo.jpg --tris 200 --iters 2000
python python/sweep.py --counts 32 128 512      # compare triangle budgets
```

**Browser** — a live fit you can point at your own image:

```sh
cd web && npm install && npm run all && npm run serve
# open http://localhost:8080
```

## How it works

### The forward pass

Coverage is `sigmoid(sd / sigma)`, where `sd` is the signed distance to the
triangle boundary — positive inside. Distance is measured to the nearest edge
*segment*, not to the nearest edge's infinite half-plane. Half-planes are
cheaper but badly overestimate distance near corners, which would give wrong
gradient magnitudes exactly where triangles meet.

Triangles composite back-to-front with alpha-over. Rendering happens in linear
light; gamma is applied only when writing a PNG.

### The backward pass

`render_with_tape` records what the reverse sweep needs; `backward` returns the
loss and one gradient per parameter — 6 position, 3 color, 1 alpha per triangle.

```rust
let (rendered, tape) = render_with_tape(&scene, params);
let (loss, grads) = backward(&scene, params, &tape, &rendered, &target);
```

Compositing is sequential, so the reverse sweep needs the canvas as it stood
*before* each triangle was painted. Storing a full canvas per triangle costs
`K × W × H × 3` floats — 630MB for 200 triangles at 512px. "Un-compositing" by
dividing by `1 - w` avoids that but blows up as `w` approaches 1. So the tape
saves only the rectangle each triangle actually touched, and memory scales with
coverage instead.

The one derivation worth reading the comments for is the distance term. For the
nearest edge `(a, b)`, closest point `a + t(b - a)`, and unit vector `u` from
that point toward the pixel:

```
d(dist)/da = -(1 - t) * u        d(dist)/db = -t * u
```

That holds whether the projection lands inside the segment or is clamped to an
endpoint — in the interior case `t` sits at a minimum of the distance, so the
term through `dt` vanishes. Carrying a spurious `dt` term is the standard way
this gradient ends up subtly wrong near corners, and it fails quietly: nothing
crashes, fits just converge worse than they should.

`finite_difference` computes the same gradient numerically. It is far too slow
to train with; it exists so the analytic path can be checked against it, which
`cargo test` does over all 10 parameters of a two-triangle scene.

### Making it converge

Three things matter more than the rest.

**Sigma annealing.** The fit starts blurry (`sigma_start = 0.02`, roughly 4px)
and sharpens geometrically to `sigma_end`. Starting sharp is the most common way
a fit stalls — with a tight sigma, a triangle that doesn't already overlap the
region it should cover sees no gradient at all and never moves. The schedule is
geometric rather than linear because sigma is a scale: halving it means the same
thing at 0.02 as at 0.002, so equal ratios deserve equal time.

**Per-parameter learning rates.** Positions live in normalized units where the
whole canvas is 1.0 wide, so they need a much smaller rate than colors. A single
global rate either freezes geometry or flings vertices off-screen.

**Color initialization from the target.** Each triangle is seeded with the
target's color at its centroid, so the fit starts near the right palette and
spends its steps on geometry instead of rediscovering that the sky is blue.

Alpha is projected into `[1e-3, 0.999]` rather than `[0, 1]`, because the
backward pass gives a clamped alpha no gradient — a triangle that reached
exactly 0 could never come back.

## Robustness

- **Invalid configs are rejected up front** with a specific error. A zero sigma
  divides by zero deep inside the coverage function and a negative learning rate
  quietly *maximizes* the loss; both are far harder to diagnose after the fact.
- **The best scene is returned, not the last.** Annealing means late iterations
  can be marginally worse than the middle of the run.
- **Divergence is caught.** A non-finite loss stops the fit before NaN can enter
  Adam's moments, where it would never leave.
- **Non-finite parameters are sanitized** each step. NaN survives `clamp` — it
  propagates through both comparisons — so it is replaced outright.
- **Early stopping** on a configurable patience, so a converged fit doesn't burn
  its remaining budget.
- **Non-square and 1-pixel targets** are handled without cropping.
- **Both CLIs report errors and exit non-zero** rather than panicking.
- **The serializer can't emit invalid JSON** — `NaN` and `inf` are not JSON, and
  a serializer that can produce unparseable output is a trap.

## Tests

```sh
cargo test --release          # 65 Rust tests
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

- **GPU rasterizer** (wgpu compute shaders). The single biggest win: it would
  remove the CPU bottleneck from end-to-end training and make large-batch
  rendering GPU-resident.
- **Gaussian primitives** alongside triangles, closer to how 3D Gaussian
  splatting parameterizes scenes.
- **Perceptual loss** (LPIPS) in place of MSE, which over-rewards blur.
- **Adaptive triangle count** — split high-error triangles, prune invisible
  ones, rather than fixing the budget up front.

## License

MIT — see [LICENSE](LICENSE).
