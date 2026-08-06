//! End-to-end tests over the public API.
//!
//! The unit tests check pieces in isolation; these check the properties a user
//! actually depends on — that a fit improves, that a fitted scene survives a
//! round trip through disk, and that resolution independence really holds.

use diffrast::{
    backward, fit, fit_within, render, render_with_tape, scene_to_json, Canvas, FitConfig,
    RenderParams, Scene, StopReason, Triangle,
};

fn reference_scene() -> Scene {
    let mut s = Scene::new([0.10, 0.11, 0.14]);
    s.push(Triangle::new([[0.20, 0.25], [0.80, 0.30], [0.50, 0.80]], [0.90, 0.30, 0.20], 0.9))
        .push(Triangle::new([[0.30, 0.55], [0.85, 0.60], [0.60, 0.15]], [0.20, 0.60, 0.90], 0.7));
    s
}

fn reference_target(w: usize, h: usize) -> Canvas {
    render(&reference_scene(), RenderParams::new(w, h, 0.002))
}

#[test]
fn fit_reduces_loss_substantially() {
    let cfg = FitConfig { triangles: 24, iters: 400, seed: 1, ..Default::default() };
    let report = fit(&reference_target(64, 64), &cfg, |_| {}).expect("valid config");

    assert!(report.improvement() > 10.0, "only improved {}x", report.improvement());
    assert_ne!(report.stop_reason, StopReason::Diverged);
}

#[test]
fn fitted_scene_upscales_without_refitting() {
    let cfg = FitConfig { triangles: 24, iters: 300, seed: 2, ..Default::default() };
    let target_small = reference_target(64, 64);
    let report = fit(&target_small, &cfg, |_| {}).expect("valid config");

    // The same scene, rendered at 4x, should still resemble the 4x target.
    // Geometry is normalized, so this must hold without any re-optimization.
    let big = render(&report.scene, RenderParams::new(256, 256, cfg.sigma_end / 4.0));
    let loss_big = big.mse(&reference_target(256, 256));

    let baseline = Canvas::filled(256, 256, [0.5; 3]).mse(&reference_target(256, 256));
    assert!(loss_big < baseline, "upscaled fit {loss_big} was no better than flat gray {baseline}");
}

#[test]
fn exported_scene_json_is_well_formed() {
    let cfg = FitConfig { triangles: 8, iters: 50, seed: 3, ..Default::default() };
    let report = fit(&reference_target(48, 48), &cfg, |_| {}).expect("valid config");
    let json = scene_to_json(&report.scene);

    assert_eq!(json.matches("\"verts\"").count(), 8);
    assert!(json.contains("\"version\": 1"));
    assert_eq!(json.matches('{').count(), json.matches('}').count());
    assert_eq!(json.matches('[').count(), json.matches(']').count());
    assert!(!json.contains(",\n  ]"), "trailing comma would be invalid JSON");
}

#[test]
fn render_is_deterministic() {
    let scene = reference_scene();
    let p = RenderParams::new(96, 96, 0.004);
    assert_eq!(render(&scene, p).data, render(&scene, p).data);
}

#[test]
fn taped_render_matches_plain_render_at_several_sizes() {
    let scene = reference_scene();
    for (w, h) in [(32, 32), (64, 48), (17, 41)] {
        let p = RenderParams::new(w, h, 0.01);
        let (taped, _) = render_with_tape(&scene, p);
        assert_eq!(render(&scene, p).data, taped.data, "mismatch at {w}x{h}");
    }
}

#[test]
fn gradient_points_downhill() {
    // A step along the negative gradient must reduce the loss. This is the
    // property the whole optimizer rests on, checked without the optimizer.
    let p = RenderParams::new(48, 48, 0.02);
    let target = reference_target(48, 48);

    let mut scene = reference_scene();
    scene.tris[0].verts[0][0] += 0.08;
    scene.tris[0].color = [0.3, 0.3, 0.3];

    let (rendered, tape) = render_with_tape(&scene, p);
    let (loss, grads) = backward(&scene, p, &tape, &rendered, &target);

    let mut params = scene.params();
    let step = 1e-3 / grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
    for (v, g) in params.iter_mut().zip(&grads) {
        *v -= step * g;
    }
    let mut stepped = scene.clone();
    stepped.set_params(&params);

    let new_loss = render(&stepped, p).mse(&target);
    assert!(new_loss < loss, "loss rose from {loss} to {new_loss}");
}

#[test]
fn non_square_and_tiny_canvases_are_handled() {
    for (w, h) in [(1, 1), (1, 64), (64, 1), (37, 11)] {
        let target = Canvas::filled(w, h, [0.4, 0.5, 0.6]);
        let cfg = FitConfig { triangles: 4, iters: 10, patience: None, ..Default::default() };
        let report = fit(&target, &cfg, |_| {}).expect("valid config");
        assert!(report.best_loss.is_finite(), "non-finite loss at {w}x{h}");
    }
}

#[test]
fn empty_scene_renders_the_background() {
    let scene = Scene::new([0.2, 0.4, 0.6]);
    let img = render(&scene, RenderParams::new(16, 16, 0.01));
    assert!(img.data.chunks_exact(3).all(|px| px == [0.2, 0.4, 0.6]));
}

#[test]
fn png_round_trips_through_disk() {
    let dir = std::env::temp_dir().join(format!("diffrast-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("round-trip.png");

    let original = render(&reference_scene(), RenderParams::new(32, 32, 0.004));
    original.save_png(&path).expect("save");
    let loaded = Canvas::load_image(&path, 32, 32).expect("load");

    // 8-bit sRGB quantization is lossy, so this is a closeness check, not
    // equality. Anything above ~1e-4 would mean the gamma handling is wrong.
    assert!(original.mse(&loaded) < 1e-4, "round-trip error {}", original.mse(&loaded));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn loading_a_missing_file_is_an_error_not_a_panic() {
    assert!(Canvas::load_image("definitely-not-here.png", 8, 8).is_err());
}

#[test]
fn fit_within_matches_documented_behavior() {
    assert_eq!(fit_within(1920, 1080, 192), (192, 108));
    assert_eq!(fit_within(100, 100, 1000), (100, 100));
}
