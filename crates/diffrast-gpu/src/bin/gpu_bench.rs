//! CPU vs GPU comparison, and a correctness check at every size.
//!
//! Prints agreement alongside timing on purpose: a rasterizer that is fast and
//! subtly wrong is worse than one that is slow, so the two numbers belong
//! together.

use std::time::Instant;

use diffrast::grad::{
    backward as cpu_backward, backward_batch as cpu_backward_batch, render_with_tape,
};
use diffrast::raster::{render as cpu_render, RenderParams};
use diffrast::{Canvas, Scene};
use diffrast_gpu::{BackwardMode, GpuRasterizer, ReduceMode};
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

    // The batch is the unit that matters for training, and on a fast device the
    // per-dispatch overhead is most of what a single small render costs — so
    // this is where the remaining time is expected to be.
    println!(
        "\n{:<28} {:>11} {:>11} {:>11} {:>9}",
        "batched backward (128px)", "cpu rayon", "gpu 1-by-1", "gpu batched", "batch win"
    );
    for &batch in &[8usize, 32, 64] {
        let tris = 64;
        let p = RenderParams::new(128, 128, 0.01);
        let scenes: Vec<Scene> = (0..batch).map(|_| scene_of(tris)).collect();
        let targets: Vec<Canvas> =
            (0..batch).map(|_| Canvas::filled(128, 128, [0.4, 0.5, 0.6])).collect();

        let cpu_time = time(2, || cpu_backward_batch(&scenes, p, &targets));
        let loop_time = time(2, || {
            for (scene, target) in scenes.iter().zip(&targets) {
                gpu.backward(scene, p, target).expect("backward");
            }
        });
        let batch_time = time(2, || gpu.backward_many(&scenes, p, &targets).expect("batched"));

        println!(
            "{:<28} {:>10.2}ms {:>10.2}ms {:>10.2}ms {:>8.1}x",
            format!("batch {batch}, {tris} tris"),
            cpu_time * 1e3,
            loop_time * 1e3,
            batch_time * 1e3,
            loop_time / batch_time
        );
    }

    // Where the GPU actually starts winning. Batching removes per-dispatch
    // overhead but does not make the GPU unconditionally faster: a 26-core CPU
    // parallelizes across batch items nearly perfectly, and for small per-item
    // work it wins outright. The dispatch rule should come from this table
    // rather than from an assumption that "GPU is faster".
    println!("\n=== crossover: CPU rayon vs GPU batched (batch of 16) ===");
    println!(
        "{:<30} {:>11} {:>12} {:>10}  winner",
        "per-item work", "cpu rayon", "gpu batched", "gpu/cpu"
    );
    for &size in &[64usize, 128, 256] {
        for &tris in &[32usize, 128, 512] {
            let batch = 16;
            let p = RenderParams::new(size, size, 0.01);
            let scenes: Vec<Scene> = (0..batch).map(|_| scene_of(tris)).collect();
            let targets: Vec<Canvas> =
                (0..batch).map(|_| Canvas::filled(size, size, [0.4, 0.5, 0.6])).collect();

            let cpu_time = time(2, || cpu_backward_batch(&scenes, p, &targets));
            let gpu_time = time(2, || gpu.backward_many(&scenes, p, &targets).expect("batched"));
            let ratio = cpu_time / gpu_time;

            println!(
                "{:<30} {:>10.2}ms {:>11.2}ms {:>9.2}x  {}",
                format!("{size}x{size}, {tris} tris"),
                cpu_time * 1e3,
                gpu_time * 1e3,
                ratio,
                if ratio > 1.0 { "GPU" } else { "cpu" }
            );
        }
    }

    // The fix for the measured bottleneck. Contention on the gradient
    // accumulators scales with the number of contending threads, so the win
    // should be largest where pixels are many and triangles (hence accumulator
    // slots) are few.
    println!("\n=== gradient reduction: global atomics vs workgroup reduction (batch of 16) ===");
    println!("{:<30} {:>12} {:>13} {:>10}", "per-item work", "direct", "workgroup", "speedup");
    for &size in &[64usize, 128, 256] {
        for &tris in &[32usize, 128, 512] {
            let batch = 16;
            let p = RenderParams::new(size, size, 0.01);
            let scenes: Vec<Scene> = (0..batch).map(|_| scene_of(tris)).collect();
            let targets: Vec<Canvas> =
                (0..batch).map(|_| Canvas::filled(size, size, [0.4, 0.5, 0.6])).collect();

            let direct = time(2, || {
                gpu.backward_many_full(&scenes, p, &targets, ReduceMode::Direct).expect("direct")
            });
            let reduced = time(2, || {
                gpu.backward_many_full(&scenes, p, &targets, ReduceMode::Workgroup)
                    .expect("workgroup")
            });

            println!(
                "{:<30} {:>11.2}ms {:>12.2}ms {:>9.2}x",
                format!("{size}x{size}, {tris} tris"),
                direct * 1e3,
                reduced * 1e3,
                direct / reduced
            );
        }
    }

    // Where the time actually goes. Extrapolating the crossover table to zero
    // triangles left a large per-batch cost that geometry could not explain;
    // this breaks that cost into phases rather than inferring it.
    println!("\n=== phase breakdown of backward_many (batch of 16, 32 triangles) ===");
    println!(
        "{:<12} {:>9} {:>9} {:>11} {:>10} {:>9} {:>10}",
        "resolution", "pack", "alloc", "dispatch", "readback", "loss", "total"
    );
    for &size in &[64usize, 128, 256] {
        let batch = 16;
        let tris = 32;
        let p = RenderParams::new(size, size, 0.01);
        let scenes: Vec<Scene> = (0..batch).map(|_| scene_of(tris)).collect();
        let targets: Vec<Canvas> =
            (0..batch).map(|_| Canvas::filled(size, size, [0.4, 0.5, 0.6])).collect();

        // Warm up, then report the median of a few runs.
        for _ in 0..2 {
            gpu.backward_many_timed(&scenes, p, &targets).expect("timed");
        }
        let mut runs: Vec<_> = (0..5)
            .map(|_| gpu.backward_many_timed(&scenes, p, &targets).expect("timed").1)
            .collect();
        runs.sort_by(|a, b| a.total_ms().partial_cmp(&b.total_ms()).unwrap());
        let t = runs[runs.len() / 2];

        println!(
            "{:<12} {:>8.2}ms {:>8.2}ms {:>10.2}ms {:>9.2}ms {:>8.2}ms {:>9.2}ms",
            format!("{size}x{size}"),
            t.pack_ms,
            t.alloc_ms,
            t.dispatch_ms,
            t.readback_ms,
            t.loss_ms,
            t.total_ms()
        );
    }

    println!("pool after the breakdown: {:?}", gpu.pool_stats());

    println!(
        "\nnote: the breakdown waits for the queue between phases so each one\n\
         measures its own work, which makes its total an upper bound — compare\n\
         phases against each other, not this total against the tables above.\n\
         The crossover table is the one to read before choosing a device."
    );
}
