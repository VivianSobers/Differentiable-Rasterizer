//! Benchmarks for the pieces that decide whether a fit is interactive.
//!
//! Run with `cargo bench`. The numbers that matter are the per-iteration cost
//! of forward+backward (which sets how long a fit takes) and the batch scaling
//! (which sets how fast training data can be generated).

use std::hint::black_box;
use std::time::{Duration, Instant};

use diffrast::grad::{backward, backward_batch, render_with_tape};
use diffrast::raster::{render, RenderParams};
use diffrast::{Canvas, Scene};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn scene_of(n: usize) -> Scene {
    let mut rng = StdRng::seed_from_u64(42);
    Scene::random(n, [0.1, 0.1, 0.12], 0.15, &mut rng)
}

/// Time `f`, returning the mean over enough runs to be stable.
fn bench<T>(name: &str, mut f: impl FnMut() -> T) {
    // Warm up so the first-touch page faults and cache misses do not land in
    // the measurement.
    for _ in 0..3 {
        black_box(f());
    }

    let mut runs = 0;
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(400) {
        black_box(f());
        runs += 1;
    }
    let per = start.elapsed().as_secs_f64() / runs as f64;

    let unit =
        if per < 1e-3 { format!("{:.1} us", per * 1e6) } else { format!("{:.2} ms", per * 1e3) };
    println!("{name:<48} {unit:>12}   ({runs} runs)");
}

fn main() {
    println!("\n=== forward render ===");
    for &size in &[128usize, 256, 512] {
        for &tris in &[64usize, 256] {
            let scene = scene_of(tris);
            let p = RenderParams::new(size, size, 0.0015);
            bench(&format!("render {size}x{size}, {tris} triangles"), || render(&scene, p));
        }
    }

    println!("\n=== forward + backward (one fit iteration) ===");
    for &size in &[128usize, 256] {
        for &tris in &[64usize, 256] {
            let scene = scene_of(tris);
            let p = RenderParams::new(size, size, 0.01);
            let target = Canvas::filled(size, size, [0.4, 0.5, 0.6]);
            bench(&format!("fwd+bwd {size}x{size}, {tris} triangles"), || {
                let (rendered, tape) = render_with_tape(&scene, p);
                backward(&scene, p, &tape, &rendered, &target)
            });
        }
    }

    println!("\n=== tape memory ===");
    for &tris in &[64usize, 256, 1024] {
        let scene = scene_of(tris);
        let p = RenderParams::new(256, 256, 0.01);
        let (_, tape) = render_with_tape(&scene, p);
        let naive = tris * 256 * 256 * 3 * 4;
        println!(
            "{:<48} {:>8} KB  (vs {} KB storing full canvases, {:.0}x less)",
            format!("tape for {tris} triangles at 256x256"),
            tape.memory_bytes() / 1024,
            naive / 1024,
            naive as f64 / tape.memory_bytes().max(1) as f64
        );
    }

    println!("\n=== batch scaling (parallel across items) ===");
    let p = RenderParams::new(128, 128, 0.01);
    let target = Canvas::filled(128, 128, [0.4, 0.5, 0.6]);
    for &batch in &[1usize, 4, 16, 64] {
        let scenes: Vec<Scene> = (0..batch).map(|_| scene_of(64)).collect();
        let targets: Vec<Canvas> = (0..batch).map(|_| target.clone()).collect();
        bench(&format!("backward_batch of {batch} (128x128, 64 tris)"), || {
            backward_batch(&scenes, p, &targets)
        });
    }
    println!("\n(divide batch timings by the batch size to compare per-item cost)");
}
