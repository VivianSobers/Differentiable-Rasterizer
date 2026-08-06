# Differentiable Rasterizer

A 2D triangle rasterizer that runs backwards. Give it a target image and it
gradient-descends the scene — vertex positions, colors, opacities — until the
render matches.

The trick that makes this possible is *soft* rasterization: a pixel's coverage
by a triangle is `sigmoid(signed_distance / sigma)` rather than a hard in/out
test. A hard test has zero gradient everywhere and an undefined one exactly at
the edge, so an optimizer never learns which way to move a vertex. Softening the
silhouette gives every pixel near an edge a real derivative.

| Target (192px) | 128 triangles, fitted |
| --- | --- |
| ![target](docs/target.png) | ![fit](docs/fit.png) |

Loss drops 454x over 1200 iterations in ~27s on a single core. The fit runs at
192px; the result is exported at 1024px without refitting, because geometry is
stored in normalized coordinates rather than pixels.

## Status

- [x] Forward rasterizer — soft coverage, alpha compositing, sRGB output
- [x] Analytic gradients w.r.t. all 10 parameters per triangle, finite-difference verified
- [x] Adam fitting loop with sigma annealing
- [ ] Python driver (targets, loss curves, GIFs)
- [ ] WASM + TypeScript viewer

## Layout

```
crates/diffrast/     Rust core
  src/scene.rs       Triangle/Scene types, flat parameter view
  src/raster.rs      Soft coverage, signed distance, forward render
  src/grad.rs        Reverse-mode gradients + finite-difference reference
  src/optim.rs       Adam with per-parameter learning rates
  src/fit.rs         Fitting loop, sigma annealing, initialization
  src/canvas.rs      f32 RGB buffer, MSE loss, PNG + sRGB conversion
  src/bin/render.rs  Demo scene renderer
  src/bin/fit.rs     Fit CLI
```

## Gradients

`render_with_tape` renders while recording what the backward pass needs;
`backward` then returns the loss and one gradient per parameter.

```rust
let (rendered, tape) = render_with_tape(&scene, params);
let (loss, grads) = backward(&scene, params, &tape, &rendered, &target);
```

Compositing is sequential, so the reverse sweep needs the canvas as it stood
before each triangle was painted. Storing a full canvas per triangle is
wasteful and "un-compositing" by dividing by `1 - w` blows up as `w` approaches
1, so the tape instead saves only the rectangle each triangle actually touches —
memory scales with coverage rather than `triangles * canvas`.

The one derivation worth reading the comments for is the distance term. For the
nearest edge `(a, b)`, closest point `a + t(b - a)`, and unit vector `u` from
that point toward the pixel:

```
d(dist)/da = -(1 - t) * u        d(dist)/db = -t * u
```

That holds whether the projection lands inside the segment or is clamped to an
endpoint — in the interior case `t` sits at a minimum of the distance, so the
term through `dt` vanishes. Carrying a spurious `dt` term is the standard way
this gradient ends up subtly wrong near corners.

`finite_difference` computes the same gradient numerically. It is far too slow
to train with; it exists so the analytic path can be checked against it, which
`cargo test` does over all 10 parameters of a two-triangle scene.

## Running

```sh
cargo test --release
cargo run --release --bin render -- out/render.png 512 0.0015
```

Arguments are `[output path] [size] [sigma]`.

Fitting an image:

```sh
cargo run --release --bin fit -- photo.jpg --tris 200 --iters 2000 --save-every 10
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tris` | 128 | triangles in the scene |
| `--iters` | 1500 | optimizer steps |
| `--size` | 192 | resolution the fit runs at |
| `--out` | `out` | output directory |
| `--save-every` | 0 (off) | write a frame every N iterations |
| `--export` | 1024 | resolution of the final render |
| `--seed` | 0 | RNG seed |

Omit the target path to fit a generated synthetic image. Outputs are `fit.png`,
`target.png`, and `loss.csv`.

## Fitting notes

Three things matter more than the rest:

**Sigma annealing.** The fit starts blurry (`sigma_start = 0.02`, roughly 4px)
and sharpens geometrically to `sigma_end`. Starting sharp is the most common way
a fit stalls — with a tight sigma, a triangle that doesn't already overlap the
region it should cover sees no gradient at all and never moves. The schedule is
geometric rather than linear because sigma is a scale: halving it means the same
thing at 0.02 as at 0.002, so equal ratios deserve equal time.

**Per-parameter learning rates.** Positions live in normalized units where the
whole canvas is 1.0 wide, so they need a much smaller rate than colors — a
single global rate either freezes geometry or flings vertices off-screen.

**Color initialization from the target.** Each triangle is seeded with the
target's color at its centroid. This is worth a surprising amount: the fit
starts near the right palette and spends its steps on geometry instead of
rediscovering that the sky is blue.

Alpha is projected into `[1e-3, 0.999]` rather than `[0, 1]`, because the
backward pass gives a clamped alpha no gradient — a triangle that reached
exactly 0 could never come back.

## Conventions

Geometry lives in normalized image space, `(0,0)` top-left to `(1,1)`
bottom-right, so a scene fitted at 128px re-renders at 2048px unchanged. Color
is linear during rendering; gamma is applied only when writing a PNG. Triangles
composite back-to-front — index 0 is furthest back.

`sigma` is the softness radius in normalized units, and it is the most important
knob in the system. Too small and gradients vanish a pixel from the edge, so
vertices never move; too large and every triangle is a smudge. Annealing it
downward over a fit is the standard approach.
