//! The fitting loop: gradient descent on a scene until its render matches a
//! target image.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::canvas::Canvas;
use crate::grad::{backward, render_with_tape};
use crate::optim::Adam;
use crate::raster::RenderParams;
use crate::scene::{Scene, Triangle};

/// Why a fit stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Ran the full iteration budget.
    Completed,
    /// The loss stopped improving for `patience` iterations.
    Converged,
    /// The loss became NaN or infinite. The returned scene is the last one
    /// known to be finite.
    Diverged,
}

/// A configuration that could not produce a sensible fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    NoTriangles,
    NonPositiveSigma,
    SigmaNotDecreasing,
    NonPositiveLearningRate,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::NoTriangles => "triangles must be greater than zero",
            Self::NonPositiveSigma => "sigma_start and sigma_end must be positive and finite",
            Self::SigmaNotDecreasing => "sigma_start must be greater than or equal to sigma_end",
            Self::NonPositiveLearningRate => "learning rates must be positive and finite",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Debug)]
pub struct FitConfig {
    pub triangles: usize,
    pub iters: usize,
    /// Softness at the start of the fit — deliberately blurry.
    pub sigma_start: f32,
    /// Softness at the end — near pixel-sharp.
    pub sigma_end: f32,
    pub lr_pos: f32,
    pub lr_color: f32,
    pub lr_alpha: f32,
    pub seed: u64,
    /// Stop early if the loss has not improved by at least `min_delta` over
    /// this many iterations. `None` disables the check.
    pub patience: Option<usize>,
    /// Improvement below this counts as no improvement.
    pub min_delta: f32,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            triangles: 128,
            iters: 1500,
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
            patience: Some(250),
            min_delta: 1e-7,
        }
    }
}

impl FitConfig {
    /// Reject configurations that cannot produce a sensible fit.
    ///
    /// Checked up front rather than left to surface as a silent non-result:
    /// a zero sigma divides by zero deep inside the coverage function, and a
    /// negative learning rate quietly maximizes the loss instead of minimizing
    /// it — both are far harder to diagnose after the fact than at the door.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.triangles == 0 {
            return Err(ConfigError::NoTriangles);
        }
        if !(self.sigma_start.is_finite() && self.sigma_end.is_finite())
            || self.sigma_start <= 0.0
            || self.sigma_end <= 0.0
        {
            return Err(ConfigError::NonPositiveSigma);
        }
        if self.sigma_start < self.sigma_end {
            return Err(ConfigError::SigmaNotDecreasing);
        }
        for lr in [self.lr_pos, self.lr_color, self.lr_alpha] {
            if !lr.is_finite() || lr <= 0.0 {
                return Err(ConfigError::NonPositiveLearningRate);
            }
        }
        Ok(())
    }
}

/// Snapshot handed to the progress callback each iteration.
pub struct FitProgress<'a> {
    pub iter: usize,
    pub loss: f32,
    pub sigma: f32,
    pub scene: &'a Scene,
}

#[derive(Clone, Debug)]
pub struct FitReport {
    /// The best scene seen, not necessarily the last one. Sigma annealing means
    /// late iterations can be slightly worse than the middle of the run, so
    /// returning the final scene would sometimes throw away the better result.
    pub scene: Scene,
    /// Loss at every iteration, in order.
    pub losses: Vec<f32>,
    pub stop_reason: StopReason,
    /// Iteration at which `scene` was captured.
    pub best_iter: usize,
    pub best_loss: f32,
}

impl FitReport {
    pub fn final_loss(&self) -> f32 {
        self.losses.last().copied().unwrap_or(f32::NAN)
    }

    pub fn initial_loss(&self) -> f32 {
        self.losses.first().copied().unwrap_or(f32::NAN)
    }

    /// How many times smaller the best loss is than the first. `1.0` means no
    /// progress; `NaN` if the fit never ran.
    pub fn improvement(&self) -> f32 {
        self.initial_loss() / self.best_loss
    }
}

/// Fit a scene of soft triangles to `target`.
///
/// The fit runs at the target's own resolution, so non-square images work
/// without cropping. `on_progress` is called once per iteration — use it to
/// save frames, print, or log. Pass `|_| {}` to run silently.
///
/// Returns `Err` only for an invalid configuration; anything that goes wrong
/// during the run itself is reported through [`FitReport::stop_reason`].
pub fn fit(
    target: &Canvas,
    cfg: &FitConfig,
    mut on_progress: impl FnMut(FitProgress),
) -> Result<FitReport, ConfigError> {
    cfg.validate()?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut scene = seed_scene(target, cfg, &mut rng);

    let mut params = scene.params();
    let lr = learning_rates(cfg, scene.len());
    let mut adam = Adam::new(params.len());
    let mut losses = Vec::with_capacity(cfg.iters);

    let mut best = (0usize, f32::INFINITY, scene.clone());
    let mut since_improved = 0usize;
    let mut stop_reason = StopReason::Completed;

    for i in 0..cfg.iters {
        let sigma = sigma_at(cfg, i);
        let rp = RenderParams::new(target.width, target.height, sigma);

        let (rendered, tape) = render_with_tape(&scene, rp);
        let (loss, grads) = backward(&scene, rp, &tape, &rendered, target);

        // Bail before a non-finite loss can poison the optimizer state: once a
        // NaN enters Adam's moments it never leaves, and every subsequent
        // parameter silently becomes NaN too.
        if !loss.is_finite() {
            stop_reason = StopReason::Diverged;
            break;
        }

        losses.push(loss);
        if loss + cfg.min_delta < best.1 {
            best = (i, loss, scene.clone());
            since_improved = 0;
        } else {
            since_improved += 1;
        }

        on_progress(FitProgress { iter: i, loss, sigma, scene: &scene });

        if cfg.patience.is_some_and(|p| since_improved >= p) {
            stop_reason = StopReason::Converged;
            break;
        }

        adam.step(&mut params, &grads, &lr);
        project(&mut params);
        scene.set_params(&params);
    }

    let (best_iter, best_loss, best_scene) = best;
    Ok(FitReport { scene: best_scene, losses, stop_reason, best_iter, best_loss })
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
    [(sum[0] / n as f64) as f32, (sum[1] / n as f64) as f32, (sum[2] / n as f64) as f32]
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
        // A single non-finite parameter would spread through the render into
        // every gradient. Clamping handles infinities; NaN survives `clamp`
        // (it propagates through both comparisons), so it needs replacing
        // outright before the clamps below can do their job.
        for v in tri.iter_mut() {
            if v.is_nan() {
                *v = 0.5;
            }
        }
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
            .push(Triangle::new(
                [[0.30, 0.55], [0.85, 0.60], [0.60, 0.15]],
                [0.20, 0.60, 0.90],
                0.7,
            ));
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

    fn target_at(size: usize) -> Canvas {
        render(&target_scene(), RenderParams::new(size, size, 0.002))
    }

    #[test]
    fn loss_decreases_on_a_synthetic_target() {
        let cfg = FitConfig { triangles: 16, iters: 200, seed: 7, ..Default::default() };
        let report = fit(&target_at(48), &cfg, |_| {}).unwrap();

        let first = report.initial_loss();
        assert!(
            report.best_loss < first * 0.5,
            "loss barely moved: {first} -> {}",
            report.best_loss
        );
        assert!(report.best_loss.is_finite(), "loss diverged");
        assert!(report.improvement() > 2.0);
    }

    #[test]
    fn fitting_is_deterministic_for_a_fixed_seed() {
        let cfg = FitConfig { triangles: 8, iters: 30, seed: 42, ..Default::default() };
        let target = target_at(32);

        let a = fit(&target, &cfg, |_| {}).unwrap();
        let b = fit(&target, &cfg, |_| {}).unwrap();
        assert_eq!(a.losses, b.losses);
        assert_eq!(a.scene.params(), b.scene.params());
    }

    #[test]
    fn progress_callback_reports_every_iteration() {
        let cfg = FitConfig { triangles: 4, iters: 10, patience: None, ..Default::default() };
        let target = Canvas::filled(32, 32, [0.5, 0.2, 0.7]);

        let mut seen = Vec::new();
        fit(&target, &cfg, |p| seen.push(p.iter)).unwrap();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn non_square_targets_are_fitted_without_cropping() {
        let cfg = FitConfig { triangles: 8, iters: 20, ..Default::default() };
        let target = Canvas::filled(64, 24, [0.3, 0.6, 0.2]);

        let report = fit(&target, &cfg, |_| {}).unwrap();
        let out = render(&report.scene, RenderParams::new(64, 24, 0.002));
        assert_eq!(out.width, 64);
        assert_eq!(out.height, 24);
        assert!(report.best_loss.is_finite());
    }

    #[test]
    fn best_scene_is_returned_not_the_last() {
        let cfg = FitConfig { triangles: 12, iters: 150, seed: 3, ..Default::default() };
        let target = target_at(40);

        let report = fit(&target, &cfg, |_| {}).unwrap();
        assert!(report.best_loss <= report.final_loss());
        assert_eq!(report.best_loss, report.losses[report.best_iter]);
    }

    #[test]
    fn patience_stops_a_stalled_fit_early() {
        // An already-perfect fit cannot improve, so patience should fire well
        // before the iteration budget runs out.
        let cfg = FitConfig { triangles: 4, iters: 1000, patience: Some(5), ..Default::default() };
        let target = Canvas::filled(24, 24, [0.4, 0.4, 0.4]);

        let report = fit(&target, &cfg, |_| {}).unwrap();
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert!(report.losses.len() < 1000, "ran {} iters", report.losses.len());
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let target = Canvas::filled(16, 16, [0.5; 3]);
        let cases = [
            (FitConfig { triangles: 0, ..Default::default() }, ConfigError::NoTriangles),
            (FitConfig { sigma_end: 0.0, ..Default::default() }, ConfigError::NonPositiveSigma),
            (
                FitConfig { sigma_start: f32::NAN, ..Default::default() },
                ConfigError::NonPositiveSigma,
            ),
            (
                FitConfig { sigma_start: 0.001, sigma_end: 0.5, ..Default::default() },
                ConfigError::SigmaNotDecreasing,
            ),
            (
                FitConfig { lr_pos: -1.0, ..Default::default() },
                ConfigError::NonPositiveLearningRate,
            ),
        ];
        for (cfg, expected) in cases {
            assert_eq!(fit(&target, &cfg, |_| {}).unwrap_err(), expected);
        }
    }

    #[test]
    fn projection_replaces_non_finite_parameters() {
        let mut p = vec![f32::NAN, f32::INFINITY, 0.5, 0.5, 0.5, 0.5, f32::NAN, 0.2, 0.3, f32::NAN];
        project(&mut p);
        assert!(p.iter().all(|v| v.is_finite()), "got {p:?}");
    }

    #[test]
    fn zero_iterations_is_not_an_error() {
        let cfg = FitConfig { triangles: 4, iters: 0, ..Default::default() };
        let report = fit(&Canvas::filled(16, 16, [0.5; 3]), &cfg, |_| {}).unwrap();
        assert!(report.losses.is_empty());
        assert!(report.final_loss().is_nan());
        assert_eq!(report.stop_reason, StopReason::Completed);
    }
}
