use crate::canvas::Canvas;
use crate::scene::{Scene, Triangle};

#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    pub width: usize,
    pub height: usize,
    /// Edge softness, in normalized image units. Larger values blur the
    /// silhouette and widen the band where gradients are non-negligible.
    ///
    /// This is the single most important knob in the whole system: too small
    /// and the gradient vanishes a pixel away from the edge so vertices never
    /// move; too large and every triangle is a smudge. Annealing it downward
    /// over a fit is the standard trick.
    pub sigma: f32,
}

impl RenderParams {
    pub fn new(width: usize, height: usize, sigma: f32) -> Self {
        Self { width, height, sigma }
    }

    /// Beyond this many sigmas the sigmoid is within ~1e-4 of saturation, so
    /// pixels further out can be skipped without visibly changing the image.
    #[inline]
    fn cull_radius(&self) -> f32 {
        6.0 * self.sigma
    }
}

/// Render a scene to a canvas, compositing triangles back-to-front.
pub fn render(scene: &Scene, p: RenderParams) -> Canvas {
    let mut canvas = Canvas::filled(p.width, p.height, scene.background);
    for tri in &scene.tris {
        composite(&mut canvas, tri, p);
    }
    canvas
}

/// Alpha-over a single triangle onto the canvas.
fn composite(canvas: &mut Canvas, tri: &Triangle, p: RenderParams) {
    let Some((x0, y0, x1, y1)) = pixel_bounds(tri, p) else { return };
    let alpha = tri.alpha.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return;
    }

    for y in y0..y1 {
        for x in x0..x1 {
            let pt = pixel_center(x, y, p);
            let w = alpha * coverage(tri, pt, p.sigma);
            if w <= 1e-6 {
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
}

/// Soft coverage of `pt` by `tri`: a sigmoid of the signed distance to the
/// triangle boundary, positive inside. Smooth in both `pt` and the vertices,
/// which is exactly what the backward pass needs.
#[inline]
pub fn coverage(tri: &Triangle, pt: [f32; 2], sigma: f32) -> f32 {
    sigmoid(signed_distance(&tri.verts, pt) / sigma.max(1e-8))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Branch on sign to avoid overflow in exp for large |x|.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Signed distance from `pt` to the triangle boundary; positive inside.
///
/// Magnitude is the distance to the nearest edge segment; the sign comes from a
/// separate winding test. Using the true distance (rather than, say, the
/// minimum edge half-plane value) keeps the falloff well-behaved near corners,
/// where half-plane distances badly overestimate how far outside a point is.
#[inline]
pub fn signed_distance(v: &[[f32; 2]; 3], pt: [f32; 2]) -> f32 {
    let d = (0..3)
        .map(|i| point_segment_distance(v[i], v[(i + 1) % 3], pt))
        .fold(f32::INFINITY, f32::min);

    if is_inside(v, pt) {
        d
    } else {
        -d
    }
}

/// Winding test that tolerates either vertex order: a point is inside when all
/// three edge cross-products share a sign.
#[inline]
fn is_inside(v: &[[f32; 2]; 3], pt: [f32; 2]) -> bool {
    let c0 = cross(v[0], v[1], pt);
    let c1 = cross(v[1], v[2], pt);
    let c2 = cross(v[2], v[0], pt);
    (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0) || (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0)
}

#[inline]
fn cross(a: [f32; 2], b: [f32; 2], pt: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
}

#[inline]
fn point_segment_distance(a: [f32; 2], b: [f32; 2], pt: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [pt[0] - a[0], pt[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    // Degenerate edge (coincident vertices): fall back to point distance.
    let t = if len2 <= 1e-20 { 0.0 } else { ((ap[0] * ab[0] + ap[1] * ab[1]) / len2).clamp(0.0, 1.0) };
    let dx = ap[0] - t * ab[0];
    let dy = ap[1] - t * ab[1];
    (dx * dx + dy * dy).sqrt()
}

/// Center of pixel `(x, y)` in normalized image space.
#[inline]
pub fn pixel_center(x: usize, y: usize, p: RenderParams) -> [f32; 2] {
    [(x as f32 + 0.5) / p.width as f32, (y as f32 + 0.5) / p.height as f32]
}

/// Pixel-space bounding box for a triangle, padded by the softness radius.
/// Returns `None` when the triangle is entirely off-canvas.
fn pixel_bounds(tri: &Triangle, p: RenderParams) -> Option<(usize, usize, usize, usize)> {
    let pad = p.cull_radius();
    let (mut lo_x, mut lo_y) = (f32::INFINITY, f32::INFINITY);
    let (mut hi_x, mut hi_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in &tri.verts {
        if !v[0].is_finite() || !v[1].is_finite() {
            return None;
        }
        lo_x = lo_x.min(v[0]);
        hi_x = hi_x.max(v[0]);
        lo_y = lo_y.min(v[1]);
        hi_y = hi_y.max(v[1]);
    }

    let to_px = |v: f32, n: usize| v * n as f32;
    let x0 = to_px(lo_x - pad, p.width).floor().max(0.0) as usize;
    let y0 = to_px(lo_y - pad, p.height).floor().max(0.0) as usize;
    let x1 = (to_px(hi_x + pad, p.width).ceil().max(0.0) as usize + 1).min(p.width);
    let y1 = (to_px(hi_y + pad, p.height).ceil().max(0.0) as usize + 1).min(p.height);

    (x0 < x1 && y0 < y1).then_some((x0, y0, x1, y1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tri() -> Triangle {
        Triangle::new([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]], [1.0, 0.0, 0.0], 1.0)
    }

    #[test]
    fn centroid_is_inside_and_far_corner_is_outside() {
        let t = unit_tri();
        assert!(signed_distance(&t.verts, [0.5, 0.4]) > 0.0);
        assert!(signed_distance(&t.verts, [0.0, 0.0]) < 0.0);
    }

    #[test]
    fn coverage_is_half_on_the_edge() {
        let t = unit_tri();
        // Midpoint of the bottom edge sits exactly on the boundary.
        let c = coverage(&t, [0.5, 0.2], 0.01);
        assert!((c - 0.5).abs() < 1e-4, "coverage at edge was {c}");
    }

    #[test]
    fn coverage_saturates_away_from_the_edge() {
        let t = unit_tri();
        assert!(coverage(&t, [0.5, 0.4], 0.002) > 0.999);
        assert!(coverage(&t, [0.02, 0.98], 0.002) < 1e-3);
    }

    #[test]
    fn winding_order_does_not_matter() {
        let t = unit_tri();
        let mut flipped = t.clone();
        flipped.verts.swap(0, 1);
        let pt = [0.5, 0.4];
        assert_eq!(
            signed_distance(&t.verts, pt).is_sign_positive(),
            signed_distance(&flipped.verts, pt).is_sign_positive()
        );
    }

    #[test]
    fn opaque_triangle_paints_over_background() {
        let mut scene = Scene::new([0.0, 0.0, 0.0]);
        scene.push(unit_tri());
        let img = render(&scene, RenderParams::new(64, 64, 0.002));
        let inside = img.get(32, 25);
        assert!(inside[0] > 0.99 && inside[1] < 0.01, "expected red inside, got {inside:?}");
        assert_eq!(img.get(0, 63), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn params_round_trip() {
        let mut scene = Scene::new([0.0; 3]);
        scene.push(unit_tri());
        scene.push(unit_tri());
        let p = scene.params();
        assert_eq!(p.len(), 2 * Triangle::N_PARAMS);
        let mut other = scene.clone();
        other.set_params(&p);
        assert_eq!(other.params(), p);
    }

    #[test]
    fn offscreen_triangle_is_culled() {
        let mut scene = Scene::new([0.1, 0.1, 0.1]);
        scene.push(Triangle::new([[5.0, 5.0], [6.0, 5.0], [5.5, 6.0]], [1.0; 3], 1.0));
        let img = render(&scene, RenderParams::new(32, 32, 0.01));
        assert!(img.data.iter().all(|&v| (v - 0.1).abs() < 1e-6));
    }
}
