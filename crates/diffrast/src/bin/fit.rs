//! Fits a triangle scene to a target image by gradient descent.

use std::error::Error;
use std::path::{Path, PathBuf};

use diffrast::raster::{coverage, pixel_center};
use diffrast::{
    fit, fit_within, render, scene_to_json, Canvas, FitConfig, RenderParams, StopReason, Triangle,
};

const USAGE: &str = "\
fit — reconstruct an image from soft triangles

usage: fit [target image] [options]

  --tris N          triangles in the scene        (default 128)
  --iters N         optimizer steps               (default 1500)
  --size N          longest side of the fit       (default 192)
  --out DIR         output directory              (default out)
  --save-every N    write a frame every N iters   (default 0, off)
  --export N        longest side of final render  (default 1024)
  --seed N          RNG seed                      (default 0)
  --patience N      stop after N stalled iters    (default 250, 0 disables)
  --quiet           suppress per-iteration output
  --help            show this message

With no target image, a synthetic one is generated.";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct Args {
    target: Option<String>,
    cfg: FitConfig,
    size: usize,
    out_dir: PathBuf,
    save_every: usize,
    export: usize,
    quiet: bool,
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args(std::env::args().skip(1).collect())? else {
        println!("{USAGE}");
        return Ok(());
    };

    let target = load_target(args.target.as_deref(), args.size)?;
    let out_dir = &args.out_dir;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let frames_dir = out_dir.join("frames");
    if args.save_every > 0 {
        std::fs::create_dir_all(&frames_dir)
            .map_err(|e| format!("cannot create {}: {e}", frames_dir.display()))?;
    }
    target.save_png(out_dir.join("target.png"))?;

    println!(
        "fitting {} triangles to {} at {}x{} for {} iters",
        args.cfg.triangles,
        args.target.as_deref().unwrap_or("<synthetic target>"),
        target.width,
        target.height,
        args.cfg.iters
    );

    let start = std::time::Instant::now();
    let mut frame_error = None;
    let report = fit(&target, &args.cfg, |p| {
        if !args.quiet && (p.iter % 100 == 0 || p.iter + 1 == args.cfg.iters) {
            println!("  iter {:>5}  loss {:.6}  sigma {:.5}", p.iter, p.loss, p.sigma);
        }
        if args.save_every > 0 && p.iter % args.save_every == 0 && frame_error.is_none() {
            let rp = RenderParams::new(target.width, target.height, p.sigma);
            let path = frames_dir.join(format!("frame_{:05}.png", p.iter / args.save_every));
            // Recorded rather than unwrapped: losing a frame should not throw
            // away a fit that may have been running for minutes.
            if let Err(e) = render(p.scene, rp).save_png(path) {
                frame_error = Some(e.to_string());
            }
        }
    })?;
    let elapsed = start.elapsed();

    if let Some(e) = frame_error {
        eprintln!("warning: frame export stopped early: {e}");
    }

    // Re-render large. Geometry is normalized, so the fitted scene upscales
    // without refitting — optimize small, export sharp.
    let (ew, eh) = scale_to(target.width, target.height, args.export);
    let sharp = RenderParams::new(ew, eh, args.cfg.sigma_end * target.width as f32 / ew as f32);
    render(&report.scene, sharp).save_png(out_dir.join("fit.png"))?;

    write_loss_csv(&out_dir.join("loss.csv"), &report.losses)?;
    std::fs::write(out_dir.join("scene.json"), scene_to_json(&report.scene))?;

    let note = match report.stop_reason {
        StopReason::Completed => "completed",
        StopReason::Converged => "converged early",
        StopReason::Diverged => "DIVERGED — returning best scene so far",
    };
    println!(
        "\n{note} in {elapsed:.2?} after {} iters\nloss {:.6} -> {:.6} ({:.1}x lower, best at iter {})\nwrote {}",
        report.losses.len(),
        report.initial_loss(),
        report.best_loss,
        report.improvement(),
        report.best_iter,
        out_dir.join("fit.png").display()
    );
    Ok(())
}

/// Returns `Ok(None)` when the user asked for help.
fn parse_args(args: Vec<String>) -> Result<Option<Args>, Box<dyn Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Ok(None);
    }

    let target = args.first().filter(|a| !a.starts_with('-')).cloned();
    let mut cfg = FitConfig::default();
    let mut size = 192;
    let mut out_dir = PathBuf::from("out");
    let mut save_every = 0;
    let mut export = 1024;
    let mut quiet = false;

    let mut i = if target.is_some() { 1 } else { 0 };
    while i < args.len() {
        let flag = &args[i];
        // Every flag but --quiet takes a value; missing values are reported
        // rather than silently defaulted, since a typo would otherwise run a
        // long fit with settings the user did not ask for.
        let value = || -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .filter(|v| !v.starts_with("--"))
                .ok_or_else(|| format!("{flag} requires a value"))
        };

        match flag.as_str() {
            "--quiet" => {
                quiet = true;
                i += 1;
                continue;
            }
            "--tris" => cfg.triangles = parse_num(flag, &value()?)?,
            "--iters" => cfg.iters = parse_num(flag, &value()?)?,
            "--size" => size = parse_num(flag, &value()?)?,
            "--seed" => cfg.seed = parse_num(flag, &value()?)? as u64,
            "--export" => export = parse_num(flag, &value()?)?,
            "--save-every" => save_every = parse_num(flag, &value()?)?,
            "--patience" => {
                let n: usize = parse_num(flag, &value()?)?;
                cfg.patience = (n > 0).then_some(n);
            }
            "--out" => out_dir = PathBuf::from(value()?),
            other => return Err(format!("unknown option {other}\n\n{USAGE}").into()),
        }
        i += 2;
    }

    if size == 0 {
        return Err("--size must be greater than zero".into());
    }
    if export == 0 {
        return Err("--export must be greater than zero".into());
    }
    cfg.validate()?;

    Ok(Some(Args { target, cfg, size, out_dir, save_every, export, quiet }))
}

fn parse_num(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse().map_err(|_| format!("{flag} expects a non-negative integer, got '{raw}'"))
}

fn load_target(path: Option<&str>, size: usize) -> Result<Canvas, Box<dyn Error>> {
    let Some(path) = path else { return Ok(synthetic_target(size)) };

    if !Path::new(path).exists() {
        return Err(format!("no such file: {path}").into());
    }
    // Probe the real dimensions first so the resize preserves aspect ratio
    // instead of squashing the image into a square.
    let (w, h) = image::image_dimensions(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let (tw, th) = fit_within(w as usize, h as usize, size);
    Canvas::load_image(path, tw, th).map_err(|e| format!("cannot decode {path}: {e}").into())
}

/// Scale `(w, h)` so the longest side is `target`, in either direction.
fn scale_to(w: usize, h: usize, target: usize) -> (usize, usize) {
    let scale = target as f64 / w.max(h) as f64;
    (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1))
}

fn write_loss_csv(path: &Path, losses: &[f32]) -> Result<(), Box<dyn Error>> {
    use std::fmt::Write as _;
    let mut csv = String::from("iter,loss\n");
    for (i, l) in losses.iter().enumerate() {
        let _ = writeln!(csv, "{i},{l}");
    }
    std::fs::write(path, csv).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(())
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

    // Composited by hand rather than with `render`, because a Scene clears to a
    // flat background color and would wipe out the gradient underneath.
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
