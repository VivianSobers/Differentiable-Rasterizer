//! Fits a triangle scene to a target image by gradient descent.
//!
//! ```text
//! cargo run --release --bin fit -- target.png [options]
//!
//!   --tris N          triangles in the scene      (default 128)
//!   --iters N         optimizer steps             (default 1500)
//!   --size N          fit resolution              (default 192)
//!   --out DIR         output directory            (default out)
//!   --save-every N    write a frame every N iters (default 0, off)
//!   --export N        final render resolution     (default 1024)
//!   --seed N          RNG seed                    (default 0)
//! ```
//!
//! With no target path, a synthetic one is generated — useful as a smoke test
//! when you just want to watch the loop run.

use std::path::{Path, PathBuf};

use diffrast::raster::{coverage, pixel_center};
use diffrast::{fit, render, Canvas, FitConfig, RenderParams, Triangle};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let target_path = args.first().filter(|a| !a.starts_with("--")).cloned();

    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let num = |name: &str, default: usize| -> usize {
        flag(name).and_then(|v| v.parse().ok()).unwrap_or(default)
    };

    let cfg = FitConfig {
        triangles: num("--tris", 128),
        iters: num("--iters", 1500),
        size: num("--size", 192),
        seed: num("--seed", 0) as u64,
        ..Default::default()
    };
    let out_dir = PathBuf::from(flag("--out").unwrap_or_else(|| "out".to_string()));
    let save_every = num("--save-every", 0);
    let export = num("--export", 1024);

    let target = match &target_path {
        Some(path) => Canvas::load_image(path, cfg.size, cfg.size)
            .unwrap_or_else(|e| panic!("failed to load {path}: {e}")),
        None => synthetic_target(cfg.size),
    };

    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");
    let frames_dir = out_dir.join("frames");
    if save_every > 0 {
        std::fs::create_dir_all(&frames_dir).expect("failed to create frames directory");
    }
    target.save_png(out_dir.join("target.png")).expect("failed to write target");

    println!(
        "fitting {} triangles to {} at {}x{} for {} iters",
        cfg.triangles,
        target_path.as_deref().unwrap_or("<synthetic target>"),
        cfg.size,
        cfg.size,
        cfg.iters
    );

    let start = std::time::Instant::now();
    let report = fit(&target, &cfg, |p| {
        if p.iter % 100 == 0 || p.iter + 1 == cfg.iters {
            println!("  iter {:>5}  loss {:.6}  sigma {:.5}", p.iter, p.loss, p.sigma);
        }
        if save_every > 0 && p.iter % save_every == 0 {
            let rp = RenderParams::new(cfg.size, cfg.size, p.sigma);
            let path = frames_dir.join(format!("frame_{:05}.png", p.iter / save_every));
            let _ = render(p.scene, rp).save_png(path);
        }
    });
    let elapsed = start.elapsed();

    // Re-render large. Geometry is in normalized space, so the fitted scene
    // upscales without refitting — the payoff of resolution-independent
    // parameters is that you optimize small and export sharp.
    let sharp = RenderParams::new(export, export, cfg.sigma_end * cfg.size as f32 / export as f32);
    render(&report.scene, sharp)
        .save_png(out_dir.join("fit.png"))
        .expect("failed to write fit");

    write_loss_csv(&out_dir.join("loss.csv"), &report.losses);

    let first = report.losses.first().copied().unwrap_or(f32::NAN);
    println!(
        "\ndone in {:.2?}  loss {:.6} -> {:.6}  ({:.1}x lower)\nwrote {}",
        elapsed,
        first,
        report.final_loss(),
        first / report.final_loss(),
        out_dir.join("fit.png").display()
    );
}

fn write_loss_csv(path: &Path, losses: &[f32]) {
    use std::fmt::Write as _;
    let mut csv = String::from("iter,loss\n");
    for (i, l) in losses.iter().enumerate() {
        let _ = writeln!(csv, "{i},{l}");
    }
    std::fs::write(path, csv).expect("failed to write loss csv");
}

/// A target with a smooth gradient plus hard-edged shapes — the two things a
/// triangle fit finds easy and hard respectively, so the result is informative.
fn synthetic_target(size: usize) -> Canvas {
    let mut c = Canvas::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            c.set(x, y, [0.15 + 0.5 * u, 0.10 + 0.4 * v, 0.55 - 0.3 * u]);
        }
    }

    // Composite hard-edged shapes over the gradient. Done by hand rather than
    // with `render`, because a Scene clears to a flat background color and
    // would wipe out the gradient underneath.
    let shapes = [
        Triangle::new([[0.15, 0.70], [0.55, 0.68], [0.35, 0.20]], [0.95, 0.85, 0.25], 1.0),
        Triangle::new([[0.50, 0.30], [0.88, 0.45], [0.62, 0.85]], [0.10, 0.25, 0.45], 1.0),
    ];
    let rp = RenderParams::new(size, size, 0.002);
    for tri in &shapes {
        for y in 0..size {
            for x in 0..size {
                let w = coverage(tri, pixel_center(x, y, rp), rp.sigma);
                if w <= 1e-4 {
                    continue;
                }
                let dst = c.get(x, y);
                let mut out = [0.0; 3];
                for ch in 0..3 {
                    out[ch] = tri.color[ch] * w + dst[ch] * (1.0 - w);
                }
                c.set(x, y, out);
            }
        }
    }
    c
}
