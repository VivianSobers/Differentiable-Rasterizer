//! CPU vs GPU comparison, and a correctness check at every size.
//!
//! Prints agreement alongside timing on purpose: a rasterizer that is fast and
//! subtly wrong is worse than one that is slow, so the two numbers belong
//! together.

use std::time::Instant;

use diffrast::grad::{backward as cpu_backward, render_with_tape};
use diffrast::raster::{render as cpu_render, RenderParams};
use diffrast::{Canvas, Scene};
use diffrast_gpu::{BackwardMode, GpuRasterizer};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn scene_of(n: usize) -> Scene {
    let mut rng = StdRng::seed_from_u64(11);
    Scene::random(n, [0.1, 0.11, 0.14], 0.15, &mut rng)
}

fn time<T>(runs: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..2 {
        std::hint::black_box(f());
    }
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(f());
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn main() {
    let gpu = match GpuRasterizer::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            eprintln!("the CPU path still works — see `cargo bench`");
            std::process::exit(1);
        }
    };
    println!("adapter: {}\n", gpu.adapter_info());

    println!("{:<28} {:>10} {:>10} {:>9}  max diff", "forward", "cpu", "gpu", "speedup");
    for &(size, tris) in &[(128usize, 64usize), (256, 128), (512, 256), (512, 1024)] {
        let scene = scene_of(tris);
        let p = RenderParams::new(size, size, 0.0015);

        let cpu_time = time(5, || cpu_render(&scene, p));
        let gpu_time = time(5, || gpu.render(&scene, p).expect("render"));

        let expected = cpu_render(&scene, p);
        let actual = gpu.render(&scene, p).expect("render");
        let diff = expected
            .data
            .iter()
            .zip(&actual.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "{:<28} {:>9.2}ms {:>9.2}ms {:>8.1}x  {:.2e}",
            format!("{size}x{size}, {tris} tris"),
            cpu_time * 1e3,
            gpu_time * 1e3,
            cpu_time / gpu_time,
            diff
        );
    }

    println!(
        "\n{:<28} {:>10} {:>10} {:>10} {:>9}  grad rel err",
        "forward + backward", "cpu", "gpu taped", "gpu recomp", "best"
    );
    for &(size, tris) in &[(128usize, 64usize), (256, 128), (256, 512)] {
        let scene = scene_of(tris);
        let p = RenderParams::new(size, size, 0.01);
        let target = Canvas::filled(size, size, [0.4, 0.5, 0.6]);

        let tape_mb = GpuRasterizer::tape_bytes(tris, size, size) / 1_048_576;

        let cpu_time = time(3, || {
            let (rendered, tape) = render_with_tape(&scene, p);
            cpu_backward(&scene, p, &tape, &rendered, &target)
        });
        let taped_time = time(3, || {
            gpu.backward_with(&scene, p, &target, BackwardMode::Taped).expect("backward")
        });
        let gpu_time = time(3, || {
            gpu.backward_with(&scene, p, &target, BackwardMode::Recompute).expect("backward")
        });

        let (rendered, tape) = render_with_tape(&scene, p);
        let (_, cpu_grads) = cpu_backward(&scene, p, &tape, &rendered, &target);
        let (_, gpu_grads) = gpu.backward(&scene, p, &target).expect("backward");

        let norm = cpu_grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
        let err =
            cpu_grads.iter().zip(&gpu_grads).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
                / norm;

        println!(
            "{:<28} {:>9.2}ms {:>9.2}ms {:>9.2}ms {:>8.1}x  {:.2e}   (tape would be {} MB)",
            format!("{size}x{size}, {tris} tris"),
            cpu_time * 1e3,
            taped_time * 1e3,
            gpu_time * 1e3,
            cpu_time / gpu_time,
            err,
            tape_mb
        );
    }

    println!(
        "\nnote: timings include buffer upload and readback, so short renders are\n\
         dominated by transfer rather than compute."
    );
}
