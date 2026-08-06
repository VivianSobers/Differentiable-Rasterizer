//! Renders a demo scene to a PNG — the day-one sanity check that the forward
//! pass works before any gradients exist.
//!
//! Usage: `cargo run --release --bin render -- [out.png] [size] [sigma]`

use std::error::Error;

use diffrast::{render, Canvas, RenderParams, Scene, Triangle};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("usage: render [out.png] [size] [sigma]");
        return Ok(());
    }

    let out = args.first().cloned().unwrap_or_else(|| "out/render.png".to_string());
    let size: usize = match args.get(1) {
        Some(v) => v.parse().map_err(|_| format!("size must be an integer, got '{v}'"))?,
        None => 512,
    };
    let sigma: f32 = match args.get(2) {
        Some(v) => v.parse().map_err(|_| format!("sigma must be a number, got '{v}'"))?,
        None => 0.0015,
    };
    if size == 0 {
        return Err("size must be greater than zero".into());
    }
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err("sigma must be positive and finite".into());
    }

    let scene = demo_scene();
    let params = RenderParams::new(size, size, sigma);

    let start = std::time::Instant::now();
    let canvas = render(&scene, params);
    let elapsed = start.elapsed();

    if let Some(dir) = std::path::Path::new(&out).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    canvas.save_png(&out).map_err(|e| format!("cannot write {out}: {e}"))?;

    println!(
        "rendered {} triangles at {size}x{size} (sigma={sigma}) in {:.2?} -> {out}",
        scene.len(),
        elapsed
    );

    // A quick self-check that the loss the fitting loop will use behaves:
    // a scene compared against itself should score exactly zero.
    let reference: Canvas = render(&scene, params);
    println!("mse against itself: {:.3e}", canvas.mse(&reference));
    Ok(())
}

/// Overlapping translucent triangles — enough to show soft edges, alpha
/// compositing, and back-to-front ordering all at once.
fn demo_scene() -> Scene {
    let mut scene = Scene::new([0.06, 0.07, 0.09]);
    scene
        .push(Triangle::new([[0.10, 0.75], [0.60, 0.72], [0.30, 0.15]], [0.91, 0.30, 0.24], 0.85))
        .push(Triangle::new([[0.40, 0.20], [0.92, 0.35], [0.68, 0.88]], [0.20, 0.60, 0.86], 0.75))
        .push(Triangle::new([[0.20, 0.45], [0.85, 0.55], [0.50, 0.95]], [0.95, 0.77, 0.20], 0.60));
    scene
}
