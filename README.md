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
- [ ] Analytic gradients w.r.t. all 10 parameters per triangle
- [ ] Adam fitting loop
- [ ] Python driver (targets, loss curves, GIFs)
- [ ] WASM + TypeScript viewer

## Layout

```
crates/diffrast/     Rust core
  src/scene.rs       Triangle/Scene types, flat parameter view
  src/raster.rs      Soft coverage, signed distance, forward render
  src/canvas.rs      f32 RGB buffer, MSE loss, PNG + sRGB conversion
  src/bin/render.rs  Demo scene renderer
```

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
