//! Python bindings that expose the rasterizer as a differentiable operation.
//!
//! This is the piece that turns the project from a fitting tool into a
//! renderer you can *train through*. Given a batch of scene parameters, it
//! returns rendered images; given upstream image gradients, it returns
//! parameter gradients. That is exactly the contract `torch.autograd.Function`
//! expects, so a neural network can emit triangle parameters and be trained
//! end-to-end against a photometric loss.
//!
//! Everything crosses the boundary as flat `f32` numpy arrays, batch-major:
//!
//! - parameters: `(B, T, 10)` — `[x0, y0, x1, y1, x2, y2, r, g, b, a]`
//! - images:     `(B, H, W, 3)` in linear light
//!
//! The GIL is released around the compute so batch items really do run in
//! parallel; holding it would serialize the rayon pool against the interpreter.

use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use diffrast::canvas::Canvas;
use diffrast::grad::{backward, render_with_tape};
use diffrast::raster::{render, RenderParams};
use diffrast::scene::{Scene, Triangle};

const N_PARAMS: usize = 10;

/// One GPU device, created on first use and shared thereafter.
///
/// Initializing a device and compiling six shaders takes long enough that doing
/// it per call would swamp the work. `OnceLock` also means a machine with no
/// usable adapter pays the failed probe exactly once and then falls back to the
/// CPU silently.
static GPU: std::sync::OnceLock<Option<diffrast_gpu::GpuRasterizer>> = std::sync::OnceLock::new();

fn gpu() -> Option<&'static diffrast_gpu::GpuRasterizer> {
    GPU.get_or_init(|| diffrast_gpu::GpuRasterizer::new().ok()).as_ref()
}

/// Which device the last call actually used. Exposed so tests can assert the
/// routing rather than infer it from timings — `device="auto"` and the silent
/// CPU fallback both make "it got faster" an unreliable signal.
static LAST_DEVICE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn note_device(gpu_used: bool) {
    LAST_DEVICE.store(u8::from(gpu_used), std::sync::atomic::Ordering::Relaxed);
}

/// Which device a call should run on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Gpu,
    /// Pick per call from the measured crossover.
    Auto,
}

fn parse_device(name: &str) -> PyResult<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "gpu" | "cuda" => Ok(Device::Gpu),
        "auto" => Ok(Device::Auto),
        other => Err(PyValueError::new_err(format!(
            "unknown device {other:?} — expected \"cpu\", \"gpu\" or \"auto\""
        ))),
    }
}

/// Resolve a requested device into "use the GPU or not", erroring only when
/// the GPU was demanded explicitly and is unavailable.
fn resolve_device(device: Device, batch: usize, tris: usize, pixels: usize) -> PyResult<bool> {
    Ok(match device {
        Device::Cpu => false,
        Device::Gpu => {
            if gpu().is_none() {
                return Err(PyValueError::new_err(
                    "device=\"gpu\" requested but no adapter is available",
                ));
            }
            true
        }
        Device::Auto => prefer_gpu(batch, tris, pixels),
    })
}

/// Should this workload go to the GPU?
///
/// The rule comes from the measured crossover rather than a preference for the
/// GPU. Contention on the gradient accumulators means the GPU only pulls ahead
/// once there is enough geometry to spread across; below that a many-core CPU
/// parallelizing over batch items wins outright. The threshold is deliberately
/// conservative — being wrong costs a factor of two, and the CPU path is never
/// catastrophic.
fn prefer_gpu(batch: usize, tris: usize, pixels: usize) -> bool {
    if gpu().is_none() {
        return false;
    }
    // Below ~64 triangles the accumulators are too few to spread contention,
    // and small batches cannot amortize the dispatch.
    tris >= 64 && batch >= 4 && batch * tris * pixels >= 4_000_000
}

/// Rebuild a scene from one row of the parameter tensor.
fn scene_from_params(params: &[f32], background: [f32; 3]) -> Scene {
    let mut scene = Scene::new(background);
    for t in params.chunks_exact(N_PARAMS) {
        scene.push(Triangle::new(
            [[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]],
            [t[6], t[7], t[8]],
            t[9],
        ));
    }
    scene
}

fn check_sigma(sigma: f32) -> PyResult<()> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive and finite"));
    }
    Ok(())
}

/// Render a batch of scenes.
///
/// `params` is `(B, T, 10)`; returns `(B, H, W, 3)` in linear light.
#[pyfunction]
#[pyo3(signature = (params, height, width, sigma, background=(0.0, 0.0, 0.0), device="auto"))]
fn render_batch<'py>(
    py: Python<'py>,
    params: PyReadonlyArray3<'py, f32>,
    height: usize,
    width: usize,
    sigma: f32,
    background: (f32, f32, f32),
    device: &str,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let device = parse_device(device)?;
    check_sigma(sigma)?;
    if height == 0 || width == 0 {
        return Err(PyValueError::new_err("height and width must be non-zero"));
    }

    let params = params.as_array();
    let (batch, tris, per) = params.dim();
    if per != N_PARAMS {
        return Err(PyValueError::new_err(format!(
            "expected last dimension {N_PARAMS}, got {per}"
        )));
    }

    let flat: Vec<f32> = params.iter().copied().collect();
    let bg = [background.0, background.1, background.2];
    let rp = RenderParams::new(width, height, sigma);
    let stride = tris * N_PARAMS;

    // Release the GIL: without this the rayon pool below would be serialized
    // against the interpreter and the batch would run single-threaded.
    let use_gpu = resolve_device(device, batch, tris, height * width)?;

    let images: Vec<f32> = py.detach(|| {
        if use_gpu {
            let scenes: Vec<Scene> =
                flat.chunks(stride).map(|row| scene_from_params(row, bg)).collect();
            match gpu().unwrap().render_many(&scenes, rp) {
                Ok(out) => {
                    note_device(true);
                    return out.into_iter().flat_map(|c| c.data).collect();
                }
                Err(e) => eprintln!("diffrast: GPU render failed ({e}), falling back to CPU"),
            }
        }
        note_device(false);

        flat.par_chunks(stride)
            .flat_map(|row| {
                let scene = scene_from_params(row, bg);
                render(&scene, rp).data
            })
            .collect()
    });

    let arr = Array4::from_shape_vec((batch, height, width, 3), images)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Backward pass: image gradients in, parameter gradients out.
///
/// `grad_images` is `(B, H, W, 3)` — the upstream gradient of the loss with
/// respect to each rendered pixel. Returns `(B, T, 10)`.
///
/// The core's `backward` computes the MSE gradient against a target, so this
/// recovers an equivalent target from the incoming gradient: for
/// `mse = mean((r - t)^2)`, `d(mse)/dr = 2(r - t)/N`, hence
/// `t = r - grad * N / 2`. Composing that way means the chain rule is applied
/// by one well-tested code path rather than two that could drift apart.
#[pyfunction]
#[pyo3(signature = (params, grad_images, sigma, background=(0.0, 0.0, 0.0), device="auto"))]
fn backward_batch<'py>(
    py: Python<'py>,
    params: PyReadonlyArray3<'py, f32>,
    grad_images: PyReadonlyArray4<'py, f32>,
    sigma: f32,
    background: (f32, f32, f32),
    device: &str,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let device = parse_device(device)?;
    check_sigma(sigma)?;

    let params = params.as_array();
    let grads_in = grad_images.as_array();
    let (batch, tris, per) = params.dim();
    let (gb, height, width, channels) = grads_in.dim();

    if per != N_PARAMS {
        return Err(PyValueError::new_err(format!(
            "expected last parameter dimension {N_PARAMS}, got {per}"
        )));
    }
    if gb != batch {
        return Err(PyValueError::new_err(format!(
            "batch mismatch: {batch} parameter rows but {gb} gradient images"
        )));
    }
    if channels != 3 {
        return Err(PyValueError::new_err("gradient images must have 3 channels"));
    }

    let flat_params: Vec<f32> = params.iter().copied().collect();
    let flat_grads: Vec<f32> = grads_in.iter().copied().collect();
    let bg = [background.0, background.1, background.2];
    let rp = RenderParams::new(width, height, sigma);
    let param_stride = tris * N_PARAMS;
    let image_stride = height * width * 3;
    let n_pixels = image_stride as f32;

    let use_gpu = resolve_device(device, batch, tris, height * width)?;

    let out: Vec<f32> = py.detach(|| {
        if use_gpu {
            let scenes: Vec<Scene> =
                flat_params.chunks(param_stride).map(|row| scene_from_params(row, bg)).collect();
            match gpu().unwrap().backward_many_from_grad(&scenes, rp, &flat_grads) {
                Ok(out) => {
                    note_device(true);
                    return out.into_iter().flatten().collect();
                }
                Err(e) => eprintln!("diffrast: GPU backward failed ({e}), falling back to CPU"),
            }
        }
        note_device(false);

        flat_params
            .par_chunks(param_stride)
            .zip(flat_grads.par_chunks(image_stride))
            .flat_map(|(row, g_img)| {
                let scene = scene_from_params(row, bg);
                let (rendered, tape) = render_with_tape(&scene, rp);

                // Invert d(mse)/dr = 2(r - t)/N to recover the target that
                // would have produced this upstream gradient.
                let mut target = Canvas::new(width, height);
                for (i, value) in target.data.iter_mut().enumerate() {
                    *value = rendered.data[i] - g_img[i] * n_pixels / 2.0;
                }

                let (_, grads) = backward(&scene, rp, &tape, &rendered, &target);
                grads
            })
            .collect()
    });

    let arr = Array3::from_shape_vec((batch, tris, N_PARAMS), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Render and score a batch against targets in one call.
///
/// Cheaper than rendering and comparing separately when only the loss and the
/// parameter gradients are wanted: the forward tape is built once and consumed
/// immediately, so the intermediate images never cross the language boundary.
#[pyfunction]
#[pyo3(signature = (params, targets, sigma, background=(0.0, 0.0, 0.0), device="auto"))]
fn fused_loss_backward<'py>(
    py: Python<'py>,
    params: PyReadonlyArray3<'py, f32>,
    targets: PyReadonlyArray4<'py, f32>,
    sigma: f32,
    background: (f32, f32, f32),
    device: &str,
) -> PyResult<(Vec<f32>, Bound<'py, PyArray3<f32>>)> {
    let device = parse_device(device)?;
    check_sigma(sigma)?;

    let params = params.as_array();
    let targets_arr = targets.as_array();
    let (batch, tris, per) = params.dim();
    let (tb, height, width, channels) = targets_arr.dim();

    if per != N_PARAMS {
        return Err(PyValueError::new_err(format!(
            "expected last parameter dimension {N_PARAMS}, got {per}"
        )));
    }
    if tb != batch {
        return Err(PyValueError::new_err(format!(
            "batch mismatch: {batch} parameter rows but {tb} targets"
        )));
    }
    if channels != 3 {
        return Err(PyValueError::new_err("targets must have 3 channels"));
    }

    let flat_params: Vec<f32> = params.iter().copied().collect();
    let flat_targets: Vec<f32> = targets_arr.iter().copied().collect();
    let bg = [background.0, background.1, background.2];
    let rp = RenderParams::new(width, height, sigma);
    let param_stride = tris * N_PARAMS;
    let image_stride = height * width * 3;

    let use_gpu = resolve_device(device, batch, tris, height * width)?;

    let results: Vec<(f32, Vec<f32>)> = py.detach(|| {
        if use_gpu {
            let scenes: Vec<Scene> =
                flat_params.chunks(param_stride).map(|row| scene_from_params(row, bg)).collect();
            let canvases: Vec<Canvas> = flat_targets
                .chunks(image_stride)
                .map(|d| Canvas { width, height, data: d.to_vec() })
                .collect();

            // Falls back rather than failing: a transient GPU error during a
            // long training run should cost one slow step, not the run.
            match gpu().unwrap().backward_many(&scenes, rp, &canvases) {
                Ok(out) => {
                    note_device(true);
                    return out;
                }
                Err(e) => eprintln!("diffrast: GPU backward failed ({e}), falling back to CPU"),
            }
        }
        note_device(false);

        flat_params
            .par_chunks(param_stride)
            .zip(flat_targets.par_chunks(image_stride))
            .map(|(row, target_data)| {
                let scene = scene_from_params(row, bg);
                let (rendered, tape) = render_with_tape(&scene, rp);
                let target = Canvas { width, height, data: target_data.to_vec() };
                backward(&scene, rp, &tape, &rendered, &target)
            })
            .collect()
    });

    let losses: Vec<f32> = results.iter().map(|(l, _)| *l).collect();
    let grads: Vec<f32> = results.into_iter().flat_map(|(_, g)| g).collect();

    let arr = Array3::from_shape_vec((batch, tris, N_PARAMS), grads)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((losses, arr.into_pyarray(py)))
}

/// Device used by the most recent call: `"gpu"` or `"cpu"`.
#[pyfunction]
fn last_device() -> &'static str {
    if LAST_DEVICE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
        "gpu"
    } else {
        "cpu"
    }
}

/// Name of the GPU adapter in use, or `None` if there is no usable device.
#[pyfunction]
fn gpu_adapter() -> Option<String> {
    gpu().map(|g| g.adapter_info().to_string())
}

/// Number of parameters per triangle, so Python never hardcodes it.
#[pyfunction]
fn params_per_triangle() -> usize {
    N_PARAMS
}

#[pymodule]
fn diffrast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_batch, m)?)?;
    m.add_function(wrap_pyfunction!(backward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fused_loss_backward, m)?)?;
    m.add_function(wrap_pyfunction!(params_per_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_adapter, m)?)?;
    m.add_function(wrap_pyfunction!(last_device, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
