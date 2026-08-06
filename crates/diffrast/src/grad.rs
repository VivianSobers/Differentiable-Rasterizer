//! Reverse-mode gradients of the MSE loss with respect to every scene
//! parameter.
//!
//! The chain the backward pass walks, per triangle `k` and pixel:
//!
//! ```text
//! loss <- pixel color <- weight w = alpha * coverage
//!                                   <- coverage = sigmoid(sd / sigma)
//!                                                 <- sd = signed distance
//!                                                         <- vertices
//! ```
//!
//! Compositing is sequential, so the backward sweep needs the canvas as it
//! stood *before* each triangle was painted. Rather than keep a full canvas per
//! triangle (or "un-composite" by dividing by `1 - w`, which blows up as `w`
//! approaches 1), the forward pass records only the rectangle each triangle
//! actually touches. Memory then scales with total coverage instead of
//! `triangles * canvas`, which for a few hundred small triangles is a large
//! difference.

use rayon::prelude::*;

use crate::canvas::Canvas;
use crate::raster::{self, RenderParams, MIN_WEIGHT};
use crate::scene::{Scene, Triangle};

/// The canvas region a single triangle overwrote, saved as it was beforehand.
struct Patch {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    /// `(x1 - x0) * (y1 - y0) * 3` values, row-major within the rectangle.
    before: Vec<f32>,
}

/// Forward-pass record needed to run [`backward`].
pub struct Tape {
    /// One entry per triangle, in scene order. `None` when the triangle was
    /// culled and contributed nothing.
    patches: Vec<Option<Patch>>,
}

impl Tape {
    /// Bytes held by the recorded patches — worth watching when scaling up
    /// triangle count or resolution.
    pub fn memory_bytes(&self) -> usize {
        self.patches.iter().flatten().map(|p| p.before.len() * std::mem::size_of::<f32>()).sum()
    }
}

/// Render a scene while recording what the backward pass needs.
///
/// Produces the same image as [`raster::render`]; the difference is only that
/// the pre-composite state of each touched region is retained.
pub fn render_with_tape(scene: &Scene, p: RenderParams) -> (Canvas, Tape) {
    let mut canvas = Canvas::filled(p.width, p.height, scene.background);
    let mut patches = Vec::with_capacity(scene.len());

    for tri in &scene.tris {
        let alpha = tri.alpha.clamp(0.0, 1.0);
        let bounds = raster::pixel_bounds(tri, p).filter(|_| alpha > 0.0);
        let Some((x0, y0, x1, y1)) = bounds else {
            patches.push(None);
            continue;
        };

        let mut before = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
        for y in y0..y1 {
            for x in x0..x1 {
                before.extend_from_slice(&canvas.get(x, y));

                let pt = raster::pixel_center(x, y, p);
                let w = alpha * raster::coverage(tri, pt, p.sigma);
                if w <= MIN_WEIGHT {
                    continue;
                }
                let dst = canvas.get(x, y);
                let mut out = [0.0; 3];
                for ch in 0..3 {
                    out[ch] = tri.color[ch] * w + dst[ch] * (1.0 - w);
                }
                canvas.set(x, y, out);
            }
        }
        patches.push(Some(Patch { x0, y0, x1, y1, before }));
    }

    (canvas, Tape { patches })
}

/// Gradient of `mse(render(scene), target)` w.r.t. the flat parameter vector
/// from [`Scene::params`].
///
/// `rendered` and `tape` must come from the same [`render_with_tape`] call on
/// `scene`. Returns `(loss, gradient)` with one gradient entry per parameter,
/// laid out `[x0, y0, x1, y1, x2, y2, r, g, b, a]` per triangle.
pub fn backward(
    scene: &Scene,
    p: RenderParams,
    tape: &Tape,
    rendered: &Canvas,
    target: &Canvas,
) -> (f32, Vec<f32>) {
    assert_eq!(tape.patches.len(), scene.len(), "tape does not match scene");
    assert_eq!(rendered.data.len(), target.data.len(), "target size mismatch");

    let loss = rendered.mse(target);
    let mut grads = vec![0.0f32; scene.len() * Triangle::N_PARAMS];

    // d(mse)/d(output pixel).
    let scale = 2.0 / rendered.data.len() as f32;
    let mut d_canvas: Vec<f32> =
        rendered.data.iter().zip(&target.data).map(|(r, t)| scale * (r - t)).collect();

    // Front-to-back: the reverse of the compositing order.
    //
    // This loop is sequential by nature and cannot be parallelized over
    // triangles: each one both reads and attenuates `d_canvas`, so triangle k
    // must see the value triangle k+1 left behind. Rows within a triangle are
    // independent, but they are far too small to be worth a fork-join. The
    // parallelism in this crate lives one level up, in `backward_batch`, where
    // whole independent images are the unit of work.
    for (k, tri) in scene.tris.iter().enumerate().rev() {
        let Some(patch) = tape.patches[k].as_ref() else { continue };
        let g = &mut grads[k * Triangle::N_PARAMS..(k + 1) * Triangle::N_PARAMS];
        backward_one(tri, p, patch, rendered, &mut d_canvas, g);
    }

    (loss, grads)
}

/// Accumulate one triangle's gradients and attenuate `d_canvas` for the layers
/// beneath it.
fn backward_one(
    tri: &Triangle,
    p: RenderParams,
    patch: &Patch,
    rendered: &Canvas,
    d_canvas: &mut [f32],
    g: &mut [f32],
) {
    let alpha = tri.alpha.clamp(0.0, 1.0);
    // A clamped alpha is locally constant, so no gradient flows to it.
    let alpha_active = tri.alpha > 0.0 && tri.alpha < 1.0;

    let row_len = (patch.x1 - patch.x0) * 3;
    for y in patch.y0..patch.y1 {
        for x in patch.x0..patch.x1 {
            let pt = raster::pixel_center(x, y, p);
            let (cov, d_cov_d_verts) = coverage_with_grad(tri, pt, p.sigma);
            let w = alpha * cov;
            if w <= MIN_WEIGHT {
                continue;
            }

            let off = (y - patch.y0) * row_len + (x - patch.x0) * 3;
            let dst = &patch.before[off..off + 3];
            let di = rendered.idx(x, y);
            let d_out = [d_canvas[di], d_canvas[di + 1], d_canvas[di + 2]];

            // out = color * w + dst * (1 - w)
            let mut d_w = 0.0;
            for ch in 0..3 {
                g[6 + ch] += d_out[ch] * w;
                d_w += d_out[ch] * (tri.color[ch] - dst[ch]);
                // What reaches the layers underneath, attenuated by this one.
                d_canvas[di + ch] = d_out[ch] * (1.0 - w);
            }

            // w = alpha * coverage
            if alpha_active {
                g[9] += d_w * cov;
            }
            let d_cov = d_w * alpha;
            for v in 0..3 {
                g[v * 2] += d_cov * d_cov_d_verts[v][0];
                g[v * 2 + 1] += d_cov * d_cov_d_verts[v][1];
            }
        }
    }
}

/// Render and differentiate a batch of independent scenes in parallel.
///
/// This is the unit of work that actually parallelizes well. A single fit is a
/// chain of sequential steps, but training a network that *predicts* scenes
/// evaluates a whole batch of unrelated images at once — and those share
/// nothing, so they scale across cores almost linearly.
///
/// Returns one `(loss, gradients)` pair per item, in input order.
pub fn backward_batch(
    scenes: &[Scene],
    p: RenderParams,
    targets: &[Canvas],
) -> Vec<(f32, Vec<f32>)> {
    assert_eq!(scenes.len(), targets.len(), "batch size mismatch");

    scenes
        .par_iter()
        .zip(targets.par_iter())
        .map(|(scene, target)| {
            let (rendered, tape) = render_with_tape(scene, p);
            backward(scene, p, &tape, &rendered, target)
        })
        .collect()
}

/// Render a batch of scenes in parallel.
pub fn render_batch(scenes: &[Scene], p: RenderParams) -> Vec<Canvas> {
    scenes.par_iter().map(|scene| crate::raster::render(scene, p)).collect()
}

/// Coverage at `pt` together with its derivative w.r.t. each of the three
/// vertices.
///
/// The distance term is where the real subtlety lives. For the nearest edge
/// `(a, b)` with closest point `a + t(b - a)` and unit vector `u` pointing from
/// that closest point toward `pt`:
///
/// ```text
/// d(dist)/da = -(1 - t) * u        d(dist)/db = -t * u
/// ```
///
/// This holds whether the projection lands inside the segment or is clamped to
/// an endpoint. When clamped, `t` is pinned and the formula is just the
/// point-to-vertex derivative; when interior, `t` is at a minimum of the
/// distance, so the extra term through `dt` vanishes and can be dropped.
/// Getting this wrong — carrying a spurious `dt` term — is the classic way a
/// hand-derived rasterizer gradient ends up subtly incorrect near corners.
fn coverage_with_grad(tri: &Triangle, pt: [f32; 2], sigma: f32) -> (f32, [[f32; 2]; 3]) {
    let v = &tri.verts;
    let sigma = sigma.max(1e-8);

    // Nearest edge, by index, along with its closest-point parameter.
    let mut best = (0usize, 0.0f32, f32::INFINITY);
    for i in 0..3 {
        let (t, d) = closest_point_on_segment(v[i], v[(i + 1) % 3], pt);
        if d < best.2 {
            best = (i, t, d);
        }
    }
    let (edge, t, dist) = best;

    let inside = raster::signed_distance(v, pt) > 0.0;
    let sign = if inside { 1.0 } else { -1.0 };
    let cov = sigmoid(sign * dist / sigma);

    // Exactly on the boundary the distance has no well-defined direction.
    // Contributing zero is the honest choice; it is a measure-zero set and the
    // optimizer sees a valid subgradient from surrounding pixels.
    if dist < 1e-9 {
        return (cov, [[0.0; 2]; 3]);
    }

    let a = v[edge];
    let b = v[(edge + 1) % 3];
    let cp = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    let u = [(pt[0] - cp[0]) / dist, (pt[1] - cp[1]) / dist];

    // d(sigmoid)/d(sd) * d(sd)/d(dist), with sd = sign * dist.
    let chain = cov * (1.0 - cov) / sigma * sign;

    let mut d = [[0.0f32; 2]; 3];
    for ax in 0..2 {
        d[edge][ax] = chain * -(1.0 - t) * u[ax];
        d[(edge + 1) % 3][ax] += chain * -t * u[ax];
    }
    (cov, d)
}

/// Returns `(t, distance)` where `t` is the clamped projection parameter of
/// `pt` onto segment `a -> b`.
#[inline]
fn closest_point_on_segment(a: [f32; 2], b: [f32; 2], pt: [f32; 2]) -> (f32, f32) {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [pt[0] - a[0], pt[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t =
        if len2 <= 1e-20 { 0.0 } else { ((ap[0] * ab[0] + ap[1] * ab[1]) / len2).clamp(0.0, 1.0) };
    let dx = ap[0] - t * ab[0];
    let dy = ap[1] - t * ab[1];
    (t, (dx * dx + dy * dy).sqrt())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Central-difference gradient of the same loss. Orders of magnitude slower
/// than [`backward`] — this exists to check it, not to use it.
pub fn finite_difference(
    scene: &Scene,
    p: RenderParams,
    target: &Canvas,
    eps: f32,
) -> (f32, Vec<f32>) {
    let base = scene.params();
    let loss = raster::render(scene, p).mse(target);
    let mut grads = vec![0.0f32; base.len()];
    let mut probe = scene.clone();

    for i in 0..base.len() {
        let mut shifted = base.clone();

        shifted[i] = base[i] + eps;
        probe.set_params(&shifted);
        let hi = raster::render(&probe, p).mse(target);

        shifted[i] = base[i] - eps;
        probe.set_params(&shifted);
        let lo = raster::render(&probe, p).mse(target);

        grads[i] = (hi - lo) / (2.0 * eps);
    }
    (loss, grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_pair(sigma: f32) -> (Scene, Canvas, RenderParams) {
        let p = RenderParams::new(48, 48, sigma);

        let mut target_scene = Scene::new([0.1, 0.12, 0.15]);
        target_scene
            .push(Triangle::new([[0.25, 0.30], [0.80, 0.22], [0.55, 0.78]], [0.9, 0.3, 0.2], 0.8))
            .push(Triangle::new([[0.15, 0.60], [0.70, 0.65], [0.40, 0.20]], [0.2, 0.6, 0.9], 0.6));
        let target = raster::render(&target_scene, p);

        // Perturbed away from the target so the gradient is non-trivial, and
        // deliberately asymmetric so no two edges tie for "nearest".
        let mut scene = Scene::new([0.1, 0.12, 0.15]);
        scene
            .push(Triangle::new([[0.21, 0.34], [0.77, 0.26], [0.58, 0.72]], [0.7, 0.4, 0.3], 0.7))
            .push(Triangle::new([[0.18, 0.55], [0.66, 0.69], [0.44, 0.24]], [0.3, 0.5, 0.8], 0.5));

        (scene, target, p)
    }

    #[test]
    fn tape_render_matches_plain_render() {
        let (scene, _, p) = scene_pair(0.02);
        let plain = raster::render(&scene, p);
        let (taped, _) = render_with_tape(&scene, p);
        assert_eq!(plain.data, taped.data);
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let (scene, target, p) = scene_pair(0.03);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (loss, analytic) = backward(&scene, p, &tape, &rendered, &target);
        let (fd_loss, numeric) = finite_difference(&scene, p, &target, 1e-3);

        assert!((loss - fd_loss).abs() < 1e-6, "loss mismatch: {loss} vs {fd_loss}");

        // Compare against the overall gradient magnitude: parameters differ in
        // scale (positions vs. colors), and a per-element relative test would
        // be dominated by the near-zero entries.
        let norm = numeric.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-8);
        for (i, (a, n)) in analytic.iter().zip(&numeric).enumerate() {
            let err = (a - n).abs() / norm;
            assert!(err < 0.02, "param {i}: analytic {a}, numeric {n}, rel err {err}");
        }
    }

    #[test]
    fn color_gradient_is_exact_for_a_single_triangle() {
        // With one opaque triangle the color derivative is linear in coverage,
        // so this is the one case where analytic and numeric should agree very
        // tightly — a good canary for sign or scaling errors.
        let p = RenderParams::new(32, 32, 0.02);
        let tri = Triangle::new([[0.2, 0.25], [0.85, 0.35], [0.5, 0.8]], [0.6, 0.4, 0.2], 1.0);

        let mut target_scene = Scene::new([0.0; 3]);
        target_scene.push(Triangle::new(tri.verts, [0.1, 0.9, 0.5], 1.0));
        let target = raster::render(&target_scene, p);

        let mut scene = Scene::new([0.0; 3]);
        scene.push(tri);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (_, analytic) = backward(&scene, p, &tape, &rendered, &target);
        let (_, numeric) = finite_difference(&scene, p, &target, 1e-3);

        for ch in 0..3 {
            let (a, n) = (analytic[6 + ch], numeric[6 + ch]);
            assert!((a - n).abs() / n.abs().max(1e-6) < 0.01, "channel {ch}: {a} vs {n}");
        }
    }

    #[test]
    fn batch_gradients_match_sequential_ones() {
        // Parallelism must not change the numbers. Each item is independent,
        // so batching is purely a scheduling decision.
        let (scene, target, p) = scene_pair(0.02);

        let scenes: Vec<Scene> = (0..4).map(|_| scene.clone()).collect();
        let targets: Vec<Canvas> = (0..4).map(|_| target.clone()).collect();

        let batched = backward_batch(&scenes, p, &targets);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (loss, grads) = backward(&scene, p, &tape, &rendered, &target);

        assert_eq!(batched.len(), 4);
        for (b_loss, b_grads) in &batched {
            assert_eq!(*b_loss, loss);
            assert_eq!(*b_grads, grads);
        }
    }

    #[test]
    fn batch_render_matches_sequential_render() {
        let (scene, _, p) = scene_pair(0.01);
        let scenes: Vec<Scene> = (0..3).map(|_| scene.clone()).collect();

        for img in render_batch(&scenes, p) {
            assert_eq!(img.data, raster::render(&scene, p).data);
        }
    }

    #[test]
    fn batch_preserves_input_order() {
        // Rayon completes work out of order; the results must not come back
        // shuffled, or a training batch would be paired with the wrong labels.
        let p = RenderParams::new(16, 16, 0.02);
        let mut scenes = Vec::new();
        let mut targets = Vec::new();
        for i in 0..6 {
            let shade = i as f32 / 6.0;
            let mut s = Scene::new([shade, shade, shade]);
            s.push(Triangle::new([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]], [1.0, 0.0, 0.0], 0.5));
            scenes.push(s);
            targets.push(Canvas::filled(16, 16, [1.0 - shade, 0.5, 0.5]));
        }

        let batched = backward_batch(&scenes, p, &targets);
        for (i, (loss, _)) in batched.iter().enumerate() {
            let (rendered, tape) = render_with_tape(&scenes[i], p);
            let (expected, _) = backward(&scenes[i], p, &tape, &rendered, &targets[i]);
            assert_eq!(*loss, expected, "item {i} out of order");
        }
    }

    #[test]
    #[should_panic(expected = "batch size mismatch")]
    fn mismatched_batch_sizes_panic() {
        let p = RenderParams::new(8, 8, 0.02);
        backward_batch(&[Scene::new([0.0; 3])], p, &[]);
    }

    #[test]
    fn zero_loss_gives_zero_gradient() {
        let (scene, _, p) = scene_pair(0.02);
        let (rendered, tape) = render_with_tape(&scene, p);
        let (loss, grads) = backward(&scene, p, &tape, &rendered, &rendered);
        assert_eq!(loss, 0.0);
        assert!(grads.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn clamped_alpha_receives_no_gradient() {
        let p = RenderParams::new(32, 32, 0.02);
        let mut scene = Scene::new([0.0; 3]);
        scene.push(Triangle::new([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]], [1.0, 0.0, 0.0], 1.0));
        let target = Canvas::filled(32, 32, [0.0, 0.0, 1.0]);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (_, grads) = backward(&scene, p, &tape, &rendered, &target);
        assert_eq!(grads[9], 0.0, "fully opaque alpha should be pinned");
    }

    #[test]
    fn culled_triangle_contributes_no_gradient() {
        let p = RenderParams::new(32, 32, 0.02);
        let mut scene = Scene::new([0.2; 3]);
        scene.push(Triangle::new([[5.0, 5.0], [6.0, 5.0], [5.5, 6.0]], [1.0; 3], 1.0));
        let target = Canvas::filled(32, 32, [0.9; 3]);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (_, grads) = backward(&scene, p, &tape, &rendered, &target);
        assert!(grads.iter().all(|g| *g == 0.0));
        assert_eq!(tape.memory_bytes(), 0);
    }
}
