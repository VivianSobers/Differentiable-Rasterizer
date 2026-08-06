//! The fitting loop: gradient descent on a scene until its render matches a
//! target image.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::canvas::Canvas;
use crate::grad::{backward, render_with_tape};
use crate::optim::Adam;
use crate::raster::RenderParams;
use crate::scene::{Scene, Triangle};

#[derive(Clone, Debug)]
pub struct FitConfig {
    pub triangles: usize,
    pub iters: usize,
    /// Resolution the fit runs at. Geometry is resolution-independent, so a
    /// scene fitted small can be re-rendered large afterwards.
    pub size: usize,
    /// Softness at the start of the fit — deliberately blurry.
    pub sigma_start: f32,
    /// Softness at the end — near pixel-sharp.
    pub sigma_end: f32,
    pub lr_pos: f32,
    pub lr_color: f32,
    pub lr_alpha: f32,
    pub seed: u64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            triangles: 128,
            iters: 1500,
            size: 192,
            // Roughly 4 pixels of blur at 192px, annealed down to well under
            // one. Starting sharp is the most common way a fit stalls: with a
            // tight sigma, a triangle that does not already overlap the region
            // it should cover sees no gradient at all and never moves.
            sigma_start: 0.02,
            sigma_end: 0.0015,
            // Positions need a much smaller rate than colors: they are in
            // normalized units where the whole canvas is 1.0 wide, so a step of
            // 0.05 would fling a vertex across the image.
            lr_pos: 0.004,
            lr_color: 0.02,
            lr_alpha: 0.01,
            seed: 0,
        }
    }
}

/// Snapshot handed to the progress callback each iteration.
pub struct FitProgress<'a> {
    pub iter: usize,
    pub loss: f32,
    pub sigma: f32,
    pub scene: &'a Scene,
}

pub struct FitReport {
    pub scene: Scene,
    /// Loss at every iteration, in order.
    pub losses: Vec<f32>,
}

impl FitReport {
    pub fn final_loss(&self) -> f32 {
        self.losses.last().copied().unwrap_or(f32::NAN)
    }
}

/// Fit a scene of soft triangles to `target`.
///
/// `on_progress` is called once per iteration — use it to save frames, print,
/// or log. Pass `|_| {}` to run silently.
pub fn fit(target: &Canvas, cfg: &FitConfig, mut on_progress: impl FnMut(FitProgress)) -> FitReport {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut scene = seed_scene(target, cfg, &mut rng);

    let mut params = scene.params();
    let lr = learning_rates(cfg, scene.len());
    let mut adam = Adam::new(params.len());
    let mut losses = Vec::with_capacity(cfg.iters);

    for i in 0..cfg.iters {
        let sigma = sigma_at(cfg, i);
        let rp = RenderParams::new(cfg.size, cfg.size, sigma);

        let (rendered, tape) = render_with_tape(&scene, rp);
        let (loss, grads) = backward(&scene, rp, &tape, &rendered, target);

        adam.step(&mut params, &grads, &lr);
        project(&mut params);
        scene.set_params(&params);

        losses.push(loss);
        on_progress(FitProgress { iter: i, loss, sigma, scene: &scene });
    }

    FitReport { scene, losses }
}

/// Softness for iteration `i`, annealed geometrically from `sigma_start` to
/// `sigma_end`.
///
/// Geometric rather than linear because sigma acts as a scale: halving it
/// matters the same amount whether it starts at 0.02 or 0.002, so the schedule
/// should spend equal time on equal ratios.
pub fn sigma_at(cfg: &FitConfig, i: usize) -> f32 {
    if cfg.iters <= 1 {
        return cfg.sigma_end;
    }
    let t = i as f32 / (cfg.iters - 1) as f32;
    cfg.sigma_start * (cfg.sigma_end / cfg.sigma_start).powf(t)
}

/// Random triangles, each tinted with the target's color where it lands.
///
/// Seeding colors from the target rather than at random is worth a surprising
/// amount: it starts the fit near the right palette, so the optimizer spends
/// its steps on geometry instead of rediscovering that the sky is blue.
fn seed_scene(target: &Canvas, cfg: &FitConfig, rng: &mut StdRng) -> Scene {
    let background = mean_color(target);
    let mut scene = Scene::random(cfg.triangles, background, 0.18, rng);

    for tri in &mut scene.tris {
        let cx = (tri.verts[0][0] + tri.verts[1][0] + tri.verts[2][0]) / 3.0;
        let cy = (tri.verts[0][1] + tri.verts[1][1] + tri.verts[2][1]) / 3.0;
        tri.color = sample(target, cx, cy);
        tri.alpha = rng.gen_range(0.3..0.7);
    }
    scene
}

fn mean_color(c: &Canvas) -> [f32; 3] {
    let n = (c.data.len() / 3) as f32;
    let mut sum = [0.0f64; 3];
    for px in c.data.chunks_exact(3) {
        for ch in 0..3 {
            sum[ch] += px[ch] as f64;
        }
    }
    [
        (sum[0] / n as f64) as f32,
        (sum[1] / n as f64) as f32,
        (sum[2] / n as f64) as f32,
    ]
}

/// Nearest-pixel sample at normalized coordinates, clamped to the canvas.
fn sample(c: &Canvas, x: f32, y: f32) -> [f32; 3] {
    let px = ((x * c.width as f32) as isize).clamp(0, c.width as isize - 1) as usize;
    let py = ((y * c.height as f32) as isize).clamp(0, c.height as isize - 1) as usize;
    c.get(px, py)
}

/// One learning rate per parameter, grouped by what the parameter means.
fn learning_rates(cfg: &FitConfig, n_tris: usize) -> Vec<f32> {
    let mut lr = Vec::with_capacity(n_tris * Triangle::N_PARAMS);
    for _ in 0..n_tris {
        lr.extend_from_slice(&[cfg.lr_pos; 6]);
        lr.extend_from_slice(&[cfg.lr_color; 3]);
        lr.push(cfg.lr_alpha);
    }
    lr
}

/// Clamp parameters back into their valid ranges after each step.
fn project(params: &mut [f32]) {
    for tri in params.chunks_exact_mut(Triangle::N_PARAMS) {
        // Positions may wander a little off-canvas — a triangle clipped by the
        // frame edge is legitimate — but not arbitrarily far, or culled
        // triangles drift away with no gradient to pull them back.
        for v in &mut tri[0..6] {
            *v = v.clamp(-0.25, 1.25);
        }
        for c in &mut tri[6..9] {
            *c = c.clamp(0.0, 1.0);
        }
        // Held strictly inside [0, 1]: the backward pass gives a clamped alpha
        // no gradient, so a triangle that reached exactly 0 or 1 could never
        // come back.
        tri[9] = tri[9].clamp(1e-3, 0.999);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::render;

    fn target_scene() -> Scene {
        let mut s = Scene::new([0.08, 0.09, 0.12]);
        s.push(Triangle::new([[0.20, 0.25], [0.80, 0.30], [0.50, 0.80]], [0.90, 0.30, 0.20], 0.9))
            .push(Triangle::new([[0.30, 0.55], [0.85, 0.60], [0.60, 0.15]], [0.20, 0.60, 0.90], 0.7));
        s
    }

    #[test]
    fn sigma_anneals_from_start_to_end() {
        let cfg = FitConfig { iters: 100, ..Default::default() };
        assert!((sigma_at(&cfg, 0) - cfg.sigma_start).abs() < 1e-9);
        assert!((sigma_at(&cfg, 99) - cfg.sigma_end).abs() < 1e-9);
        // Monotonically decreasing.
        for i in 1..100 {
            assert!(sigma_at(&cfg, i) < sigma_at(&cfg, i - 1));
        }
    }

    #[test]
    fn single_iteration_config_uses_final_sigma() {
        let cfg = FitConfig { iters: 1, ..Default::default() };
        assert_eq!(sigma_at(&cfg, 0), cfg.sigma_end);
    }

    #[test]
    fn projection_keeps_parameters_in_range() {
        let mut p = vec![-5.0, 5.0, 0.5, 0.5, 0.5, 0.5, -1.0, 2.0, 0.5, 1.7];
        project(&mut p);
        assert_eq!(&p[0..2], &[-0.25, 1.25]);
        assert_eq!(&p[6..9], &[0.0, 1.0, 0.5]);
        assert_eq!(p[9], 0.999);
    }

    #[test]
    fn loss_decreases_on_a_synthetic_target() {
        let cfg = FitConfig { triangles: 16, iters: 200, size: 48, seed: 7, ..Default::default() };
        let target = render(&target_scene(), RenderParams::new(cfg.size, cfg.size, 0.002));

        let report = fit(&target, &cfg, |_| {});

        let first = report.losses[0];
        let last = report.final_loss();
        assert!(last < first * 0.5, "loss barely moved: {first} -> {last}");
        assert!(last.is_finite(), "loss diverged to {last}");
    }

    #[test]
    fn fitting_is_deterministic_for_a_fixed_seed() {
        let cfg = FitConfig { triangles: 8, iters: 30, size: 32, seed: 42, ..Default::default() };
        let target = render(&target_scene(), RenderParams::new(cfg.size, cfg.size, 0.002));

        let a = fit(&target, &cfg, |_| {});
        let b = fit(&target, &cfg, |_| {});
        assert_eq!(a.losses, b.losses);
        assert_eq!(a.scene.params(), b.scene.params());
    }

    #[test]
    fn progress_callback_reports_every_iteration() {
        let cfg = FitConfig { triangles: 4, iters: 10, size: 32, ..Default::default() };
        let target = Canvas::filled(cfg.size, cfg.size, [0.5, 0.2, 0.7]);

        let mut seen = Vec::new();
        fit(&target, &cfg, |p| seen.push(p.iter));
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
    }
}
