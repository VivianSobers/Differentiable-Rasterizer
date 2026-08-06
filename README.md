# Differentiable Rasterizer

A 2D triangle rasterizer that runs backwards. Give it a target image and it
gradient-descends the scene — vertex positions, colors, opacities — until the
render matches.

The trick that makes this possible is *soft* rasterization: a pixel's coverage
by a triangle is `sigmoid(signed_distance / sigma)` rather than a hard in/out
test. A hard test has zero gradient everywhere and an undefined one exactly at
the edge, so an optimizer never learns which way to move a vertex. Softening the
silhouette gives every pixel near an edge a real derivative.

## Status

- [x] Forward rasterizer — soft coverage, alpha compositing, sRGB output
- [x] Analytic gradients w.r.t. all 10 parameters per triangle, finite-difference verified
- [ ] Adam fitting loop
- [ ] Python driver (targets, loss curves, GIFs)
- [ ] WASM + TypeScript viewer

## Layout

```
crates/diffrast/     Rust core
  src/scene.rs       Triangle/Scene types, flat parameter view
  src/raster.rs      Soft coverage, signed distance, forward render
  src/grad.rs        Reverse-mode gradients + finite-difference reference
  src/canvas.rs      f32 RGB buffer, MSE loss, PNG + sRGB conversion
  src/bin/render.rs  Demo scene renderer
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

## Conventions

Geometry lives in normalized image space, `(0,0)` top-left to `(1,1)`
bottom-right, so a scene fitted at 128px re-renders at 2048px unchanged. Color
is linear during rendering; gamma is applied only when writing a PNG. Triangles
composite back-to-front — index 0 is furthest back.

`sigma` is the softness radius in normalized units, and it is the most important
knob in the system. Too small and gradients vanish a pixel from the edge, so
vertices never move; too large and every triangle is a smudge. Annealing it
downward over a fit is the standard approach.
