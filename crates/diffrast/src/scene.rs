use rand::Rng;

/// A single soft triangle.
///
/// Vertices live in normalized image space: `(0.0, 0.0)` is the top-left
/// corner, `(1.0, 1.0)` the bottom-right. Keeping geometry resolution-
/// independent means a scene fitted at 128px can be re-rendered at 2048px
/// without touching the parameters.
#[derive(Clone, Debug)]
pub struct Triangle {
    /// Vertices `[[x, y]; 3]`, counter-clockwise or clockwise — either works.
    pub verts: [[f32; 2]; 3],
    /// Linear RGB in `[0, 1]`.
    pub color: [f32; 3],
    /// Opacity in `[0, 1]`.
    pub alpha: f32,
}

impl Triangle {
    /// Flat parameter count per triangle: 6 position + 3 color + 1 alpha.
    pub const N_PARAMS: usize = 10;

    pub fn new(verts: [[f32; 2]; 3], color: [f32; 3], alpha: f32) -> Self {
        Self { verts, color, alpha }
    }
}

/// An ordered list of triangles, composited back-to-front (index 0 is furthest
/// back, so later triangles paint over earlier ones).
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub tris: Vec<Triangle>,
    /// Color the canvas is cleared to before compositing.
    pub background: [f32; 3],
}

impl Scene {
    pub fn new(background: [f32; 3]) -> Self {
        Self { tris: Vec::new(), background }
    }

    pub fn push(&mut self, t: Triangle) -> &mut Self {
        self.tris.push(t);
        self
    }

    pub fn len(&self) -> usize {
        self.tris.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// Random small triangles scattered over the canvas — the usual starting
    /// point for a fit, since starting from a degenerate or uniform scene gives
    /// the optimizer nothing to break symmetry with.
    pub fn random(n: usize, background: [f32; 3], seed_scale: f32, rng: &mut impl Rng) -> Self {
        let mut scene = Scene::new(background);
        for _ in 0..n {
            let cx: f32 = rng.gen();
            let cy: f32 = rng.gen();
            let mut vert = || {
                [
                    cx + rng.gen_range(-seed_scale..seed_scale),
                    cy + rng.gen_range(-seed_scale..seed_scale),
                ]
            };
            scene.push(Triangle::new(
                [vert(), vert(), vert()],
                [rng.gen(), rng.gen(), rng.gen()],
                rng.gen_range(0.3..0.8),
            ));
        }
        scene
    }

    /// Flatten every triangle's parameters into one contiguous vector, in the
    /// order `[x0, y0, x1, y1, x2, y2, r, g, b, a]` per triangle. The optimizer
    /// works on this flat view; [`Scene::set_params`] writes it back.
    pub fn params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.tris.len() * Triangle::N_PARAMS);
        for t in &self.tris {
            for v in &t.verts {
                out.push(v[0]);
                out.push(v[1]);
            }
            out.extend_from_slice(&t.color);
            out.push(t.alpha);
        }
        out
    }

    pub fn set_params(&mut self, p: &[f32]) {
        assert_eq!(p.len(), self.tris.len() * Triangle::N_PARAMS, "parameter length mismatch");
        for (t, chunk) in self.tris.iter_mut().zip(p.chunks_exact(Triangle::N_PARAMS)) {
            t.verts = [[chunk[0], chunk[1]], [chunk[2], chunk[3]], [chunk[4], chunk[5]]];
            t.color = [chunk[6], chunk[7], chunk[8]];
            t.alpha = chunk[9];
        }
    }
}
