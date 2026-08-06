//! GPU rasterizer: the same soft-triangle renderer and gradients, in WGSL
//! compute shaders on top of `wgpu`.
//!
//! The CPU implementation parallelizes over *images*, because a single fit is a
//! sequential chain — each triangle composites onto what the previous one left.
//! This flips the loop: one thread per pixel, each walking the whole triangle
//! list. The chain still exists, but it runs inside a thread instead of across
//! them, so a single large render saturates the device.
//!
//! Correctness is defined by agreement with the CPU path, which is checked by
//! tests rather than assumed. That includes replicating the CPU's bounding-box
//! cull exactly — see `common.wgsl` for why that matters.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use diffrast::canvas::Canvas;
use diffrast::raster::RenderParams;
use diffrast::scene::{Scene, Triangle};
use wgpu::util::DeviceExt;

/// Must match the `Params` struct in `common.wgsl`, including padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    n_tris: u32,
    sigma: f32,
    background: [f32; 3],
    min_weight: f32,
    write_tape: u32,
    _pad: [u32; 3],
}

/// Something went wrong talking to the GPU.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    DeviceRequest(String),
    Readback(String),
    /// The requested work exceeds a device limit.
    TooLarge(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(
                f,
                "no GPU adapter found — install a Vulkan/Metal/DX12 driver, or use the CPU path"
            ),
            Self::DeviceRequest(m) => write!(f, "could not create device: {m}"),
            Self::Readback(m) => write!(f, "could not read results back: {m}"),
            Self::TooLarge(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A GPU device with the rasterizer pipelines compiled and ready.
///
/// Creating this is expensive — it initializes a device and compiles shaders —
/// so build one and reuse it across renders.
pub struct GpuRasterizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    forward: wgpu::ComputePipeline,
    backward: wgpu::ComputePipeline,
    backward_recompute: wgpu::ComputePipeline,
    forward_batch: wgpu::ComputePipeline,
    backward_batch: wgpu::ComputePipeline,
    info: String,
}

/// How the backward pass recovers the canvas state beneath each triangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackwardMode {
    /// Store it during the forward pass. O(T) compute, O(T x pixels) memory.
    Taped,
    /// Recompute it on demand. O(T^2) compute, no extra memory.
    ///
    /// Usually the faster of the two despite the extra arithmetic: the tape is
    /// large enough that moving it dominates, and the bounding-box cull makes
    /// the recomputation much cheaper than its complexity suggests.
    Recompute,
}

impl GpuRasterizer {
    /// Initialize the default adapter.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let info = format!("{} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("diffrast"),
                required_features: wgpu::Features::empty(),
                // Ask for the adapter's real limits: the defaults cap storage
                // buffers at 128 MB, which the tape exceeds at quite ordinary
                // resolutions.
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let common = include_str!("shaders/common.wgsl");
        let forward = Self::pipeline(
            &device,
            "forward",
            &format!("{common}\n{}", include_str!("shaders/forward.wgsl")),
        );
        let backward = Self::pipeline(
            &device,
            "backward",
            &format!("{common}\n{}", include_str!("shaders/backward.wgsl")),
        );
        let backward_recompute = Self::pipeline(
            &device,
            "backward_recompute",
            &format!("{common}\n{}", include_str!("shaders/backward_recompute.wgsl")),
        );

        let forward_batch = Self::pipeline(
            &device,
            "forward_batch",
            &format!("{common}\n{}", include_str!("shaders/forward_batch.wgsl")),
        );
        let backward_batch = Self::pipeline(
            &device,
            "backward_batch",
            &format!("{common}\n{}", include_str!("shaders/backward_batch.wgsl")),
        );

        Ok(Self {
            device,
            queue,
            forward,
            backward,
            backward_recompute,
            forward_batch,
            backward_batch,
            info,
        })
    }

    fn pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Human-readable description of the adapter in use.
    pub fn adapter_info(&self) -> &str {
        &self.info
    }

    /// Bytes the tape needs for a given workload — worth checking before a
    /// large render, since it scales with `triangles * pixels`.
    pub fn tape_bytes(n_tris: usize, width: usize, height: usize) -> u64 {
        (n_tris as u64) * (width as u64) * (height as u64) * 3 * 4
    }

    fn flatten(scene: &Scene) -> Vec<f32> {
        scene.params()
    }

    fn gpu_params(scene: &Scene, p: RenderParams, write_tape: bool) -> GpuParams {
        GpuParams {
            width: p.width as u32,
            height: p.height as u32,
            n_tris: scene.len() as u32,
            sigma: p.sigma,
            background: scene.background,
            // Must match `raster::MIN_WEIGHT` on the CPU side.
            min_weight: 1e-6,
            write_tape: write_tape as u32,
            _pad: [0; 3],
        }
    }

    /// Render a scene on the GPU.
    pub fn render(&self, scene: &Scene, p: RenderParams) -> Result<Canvas, GpuError> {
        let (canvas, _) = self.render_inner(scene, p, false)?;
        Ok(canvas)
    }

    fn render_inner(
        &self,
        scene: &Scene,
        p: RenderParams,
        keep_tape: bool,
    ) -> Result<(Canvas, Option<wgpu::Buffer>), GpuError> {
        let n_tris = scene.len().max(1);
        let pixels = p.width * p.height;

        if keep_tape {
            let needed = Self::tape_bytes(n_tris, p.width, p.height);
            let limit = self.device.limits().max_storage_buffer_binding_size;
            if needed > limit {
                return Err(GpuError::TooLarge(format!(
                    "tape needs {} MB but the device caps storage buffers at {} MB — \
                     render in tiles or reduce triangles/resolution",
                    needed / 1_048_576,
                    limit / 1_048_576
                )));
            }
        }

        let tris = Self::flatten(scene);
        // An empty scene still needs a non-zero binding.
        let tri_data = if tris.is_empty() { vec![0.0f32; Triangle::N_PARAMS] } else { tris };

        let params = Self::gpu_params(scene, p, keep_tape);
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tri_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("triangles"),
            contents: bytemuck::cast_slice(&tri_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image"),
            size: (pixels * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tape_size = if keep_tape { Self::tape_bytes(n_tris, p.width, p.height) } else { 4 };
        let tape_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tape"),
            size: tape_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward"),
            layout: &self.forward.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tape_buf.as_entire_binding() },
            ],
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(p.width.div_ceil(8) as u32, p.height.div_ceil(8) as u32, 1);
        }

        let data = self.read_buffer(&mut encoder, &out_buf, (pixels * 3 * 4) as u64)?;
        let canvas = Canvas { width: p.width, height: p.height, data };
        Ok((canvas, if keep_tape { Some(tape_buf) } else { None }))
    }

    /// Render and compute gradients in one round trip.
    ///
    /// Returns `(loss, gradients)` laid out exactly like the CPU
    /// `diffrast::backward` — 10 values per triangle.
    pub fn backward(
        &self,
        scene: &Scene,
        p: RenderParams,
        target: &Canvas,
    ) -> Result<(f32, Vec<f32>), GpuError> {
        self.backward_with(scene, p, target, BackwardMode::Recompute)
    }

    /// Gradients, choosing explicitly how the canvas state is recovered.
    ///
    /// `Recompute` is the default because it wins on every device measured so
    /// far, but `Taped` is kept: its O(T) compute should overtake on a device
    /// with enough bandwidth and a modest triangle count, and having both makes
    /// that a measurement rather than a guess.
    pub fn backward_with(
        &self,
        scene: &Scene,
        p: RenderParams,
        target: &Canvas,
        mode: BackwardMode,
    ) -> Result<(f32, Vec<f32>), GpuError> {
        assert_eq!(target.width, p.width, "target width mismatch");
        assert_eq!(target.height, p.height, "target height mismatch");

        let taped = mode == BackwardMode::Taped;
        let (rendered, tape_buf) = self.render_inner(scene, p, taped)?;

        let n_tris = scene.len();
        if n_tris == 0 {
            return Ok((rendered.mse(target), Vec::new()));
        }

        let params = Self::gpu_params(scene, p, true);
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tri_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("triangles"),
            contents: bytemuck::cast_slice(&scene.params()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rendered_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rendered"),
            contents: bytemuck::cast_slice(&rendered.data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let target_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target"),
            contents: bytemuck::cast_slice(&target.data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let grad_len = n_tris * Triangle::N_PARAMS;
        // Zeroed: the shader accumulates into these with atomic adds.
        let grad_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grads"),
            contents: bytemuck::cast_slice(&vec![0.0f32; grad_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let pipeline = if taped { &self.backward } else { &self.backward_recompute };
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: rendered_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: target_buf.as_entire_binding() },
        ];
        if taped {
            let tape_buf = tape_buf.as_ref().expect("tape requested");
            entries
                .push(wgpu::BindGroupEntry { binding: 4, resource: tape_buf.as_entire_binding() });
            entries
                .push(wgpu::BindGroupEntry { binding: 5, resource: grad_buf.as_entire_binding() });
        } else {
            entries
                .push(wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() });
        }

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backward"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(p.width.div_ceil(8) as u32, p.height.div_ceil(8) as u32, 1);
        }

        let grads = self.read_buffer(&mut encoder, &grad_buf, (grad_len * 4) as u64)?;
        Ok((rendered.mse(target), grads))
    }

    /// Render a batch of scenes in a single dispatch.
    ///
    /// Every scene must have the same triangle count. Returns one canvas per
    /// scene, in input order.
    ///
    /// Worth preferring over calling [`Self::render`] in a loop: measurement on
    /// a 4090 shows small renders are dominated by per-dispatch overhead rather
    /// than arithmetic, so amortizing that across a batch is where the time
    /// actually goes.
    pub fn render_many(&self, scenes: &[Scene], p: RenderParams) -> Result<Vec<Canvas>, GpuError> {
        let Some(n_tris) = self.check_batch(scenes)? else {
            return Ok(Vec::new());
        };

        let batch = scenes.len();
        let pixels = p.width * p.height;
        let (tri_buf, bg_buf) = self.pack_scenes(scenes);

        let mut params = Self::gpu_params(&scenes[0], p, false);
        params.n_tris = n_tris as u32;
        let param_buf = self.uniform(&params);

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("images"),
            size: (batch * pixels * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward_batch"),
            layout: &self.forward_batch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bg_buf.as_entire_binding() },
            ],
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_batch);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                p.width.div_ceil(8) as u32,
                p.height.div_ceil(8) as u32,
                batch as u32,
            );
        }

        let data = self.read_buffer(&mut encoder, &out_buf, (batch * pixels * 3 * 4) as u64)?;
        Ok(data
            .chunks_exact(pixels * 3)
            .map(|chunk| Canvas { width: p.width, height: p.height, data: chunk.to_vec() })
            .collect())
    }

    /// Render and differentiate a batch in a single pair of dispatches.
    ///
    /// Returns one `(loss, gradients)` per scene, matching the CPU
    /// `diffrast::grad::backward_batch` exactly.
    pub fn backward_many(
        &self,
        scenes: &[Scene],
        p: RenderParams,
        targets: &[Canvas],
    ) -> Result<Vec<(f32, Vec<f32>)>, GpuError> {
        if scenes.len() != targets.len() {
            return Err(GpuError::TooLarge(format!(
                "batch mismatch: {} scenes but {} targets",
                scenes.len(),
                targets.len()
            )));
        }
        let Some(n_tris) = self.check_batch(scenes)? else {
            return Ok(Vec::new());
        };

        let batch = scenes.len();

        let rendered = self.render_many(scenes, p)?;

        let (tri_buf, bg_buf) = self.pack_scenes(scenes);
        let mut params = Self::gpu_params(&scenes[0], p, false);
        params.n_tris = n_tris as u32;
        let param_buf = self.uniform(&params);

        let flat_rendered: Vec<f32> =
            rendered.iter().flat_map(|c| c.data.iter().copied()).collect();
        let flat_targets: Vec<f32> = targets.iter().flat_map(|c| c.data.iter().copied()).collect();

        let rendered_buf = self.storage("rendered", &flat_rendered);
        let target_buf = self.storage("targets", &flat_targets);
        let grad_len = batch * n_tris * Triangle::N_PARAMS;
        let grad_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grads"),
            contents: bytemuck::cast_slice(&vec![0.0f32; grad_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backward_batch"),
            layout: &self.backward_batch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rendered_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: target_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bg_buf.as_entire_binding() },
            ],
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backward_batch);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                p.width.div_ceil(8) as u32,
                p.height.div_ceil(8) as u32,
                batch as u32,
            );
        }

        let grads = self.read_buffer(&mut encoder, &grad_buf, (grad_len * 4) as u64)?;
        let stride = n_tris * Triangle::N_PARAMS;

        Ok(rendered
            .iter()
            .zip(targets)
            .zip(grads.chunks_exact(stride))
            .map(|((r, t), g)| (r.mse(t), g.to_vec()))
            .collect())
    }

    /// Validates a batch and returns its shared triangle count.
    /// `Ok(None)` means the batch was empty.
    fn check_batch(&self, scenes: &[Scene]) -> Result<Option<usize>, GpuError> {
        let Some(first) = scenes.first() else { return Ok(None) };
        let n_tris = first.len();
        if n_tris == 0 {
            return Err(GpuError::TooLarge("scenes must contain at least one triangle".into()));
        }
        // A ragged batch would silently read another scene's triangles rather
        // than fail, so it is rejected up front.
        if let Some(bad) = scenes.iter().position(|s| s.len() != n_tris) {
            return Err(GpuError::TooLarge(format!(
                "batched scenes must share a triangle count: item 0 has {n_tris}, item {bad} has {}",
                scenes[bad].len()
            )));
        }
        Ok(Some(n_tris))
    }

    /// Pack a batch's triangles and backgrounds into two buffers.
    fn pack_scenes(&self, scenes: &[Scene]) -> (wgpu::Buffer, wgpu::Buffer) {
        let tris: Vec<f32> = scenes.iter().flat_map(|s| s.params()).collect();
        let backgrounds: Vec<f32> = scenes.iter().flat_map(|s| s.background).collect();
        (self.storage("triangles", &tris), self.storage("backgrounds", &backgrounds))
    }

    fn storage(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn uniform(&self, params: &GpuParams) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Copy a storage buffer back to the host as `f32`s.
    fn read_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        size: u64,
    ) -> Result<Vec<f32>, GpuError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = std::mem::replace(
            encoder,
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
        );
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Readback(format!("{e:?}")))?;

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuError::Readback(format!("{e:?}"))),
            Err(e) => return Err(GpuError::Readback(e.to_string())),
        }

        let data = slice.get_mapped_range().map_err(|e| GpuError::Readback(format!("{e:?}")))?;
        let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffrast::grad::{backward as cpu_backward, render_with_tape};
    use diffrast::raster::render as cpu_render;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Skips rather than fails when no adapter exists, so the suite still runs
    /// on machines without a usable GPU driver.
    fn gpu() -> Option<GpuRasterizer> {
        match GpuRasterizer::new() {
            Ok(g) => {
                eprintln!("gpu: {}", g.adapter_info());
                Some(g)
            }
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn test_scene(n: usize) -> Scene {
        let mut rng = StdRng::seed_from_u64(7);
        Scene::random(n, [0.1, 0.12, 0.15], 0.2, &mut rng)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn forward_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        for (w, h, sigma, tris) in [(32, 32, 0.01, 8), (64, 48, 0.004, 24), (37, 19, 0.02, 3)] {
            let scene = test_scene(tris);
            let p = RenderParams::new(w, h, sigma);

            let expected = cpu_render(&scene, p);
            let actual = gpu.render(&scene, p).expect("render");

            // Not bit-exact: `exp` differs between the CPU's libm and the
            // device's implementation. Anything above ~1e-5 would mean a real
            // disagreement rather than rounding.
            let diff = max_abs_diff(&expected.data, &actual.data);
            assert!(diff < 1e-5, "{w}x{h} sigma {sigma}: max diff {diff}");
        }
    }

    #[test]
    fn empty_scene_renders_background() {
        let Some(gpu) = gpu() else { return };
        let scene = Scene::new([0.2, 0.4, 0.6]);
        let img = gpu.render(&scene, RenderParams::new(16, 16, 0.01)).expect("render");
        assert!(img.data.chunks_exact(3).all(|px| (px[0] - 0.2).abs() < 1e-6));
    }

    #[test]
    fn backward_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let scene = test_scene(12);
        let p = RenderParams::new(48, 48, 0.02);
        let target = Canvas::filled(48, 48, [0.5, 0.3, 0.7]);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (cpu_loss, cpu_grads) = cpu_backward(&scene, p, &tape, &rendered, &target);
        let (gpu_loss, gpu_grads) = gpu.backward(&scene, p, &target).expect("backward");

        assert!((cpu_loss - gpu_loss).abs() < 1e-6, "loss {cpu_loss} vs {gpu_loss}");
        assert_eq!(cpu_grads.len(), gpu_grads.len());

        // Compared against gradient magnitude: atomic accumulation sums in a
        // nondeterministic order, so individual values differ in their last
        // bits even when the computation is identical.
        let norm = cpu_grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
        let err = max_abs_diff(&cpu_grads, &gpu_grads) / norm;
        assert!(err < 1e-3, "relative gradient error {err}");
    }

    #[test]
    fn backward_gradient_points_downhill() {
        let Some(gpu) = gpu() else { return };
        let scene = test_scene(8);
        let p = RenderParams::new(32, 32, 0.03);
        let target = Canvas::filled(32, 32, [0.6, 0.2, 0.4]);

        let (loss, grads) = gpu.backward(&scene, p, &target).expect("backward");

        let mut params = scene.params();
        let norm = grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
        for (v, g) in params.iter_mut().zip(&grads) {
            *v -= 1e-3 / norm * g;
        }
        let mut stepped = scene.clone();
        stepped.set_params(&params);

        let new_loss = gpu.render(&stepped, p).expect("render").mse(&target);
        assert!(new_loss < loss, "loss rose from {loss} to {new_loss}");
    }

    #[test]
    fn both_backward_modes_agree() {
        // The two variants recover the canvas state by completely different
        // means — stored versus recomputed — so agreement between them is a
        // strong check that neither has drifted.
        let Some(gpu) = gpu() else { return };
        let scene = test_scene(10);
        let p = RenderParams::new(40, 40, 0.02);
        let target = Canvas::filled(40, 40, [0.3, 0.6, 0.5]);

        let (taped_loss, taped) =
            gpu.backward_with(&scene, p, &target, BackwardMode::Taped).expect("taped");
        let (recomp_loss, recomp) =
            gpu.backward_with(&scene, p, &target, BackwardMode::Recompute).expect("recompute");

        assert!((taped_loss - recomp_loss).abs() < 1e-9);
        let norm = taped.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
        let err = max_abs_diff(&taped, &recomp) / norm;
        assert!(err < 1e-3, "modes disagree by {err}");
    }

    #[test]
    fn taped_mode_matches_cpu_too() {
        let Some(gpu) = gpu() else { return };
        let scene = test_scene(12);
        let p = RenderParams::new(48, 48, 0.02);
        let target = Canvas::filled(48, 48, [0.5, 0.3, 0.7]);

        let (rendered, tape) = render_with_tape(&scene, p);
        let (_, cpu_grads) = cpu_backward(&scene, p, &tape, &rendered, &target);
        let (_, gpu_grads) =
            gpu.backward_with(&scene, p, &target, BackwardMode::Taped).expect("backward");

        let norm = cpu_grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
        assert!(max_abs_diff(&cpu_grads, &gpu_grads) / norm < 1e-3);
    }

    #[test]
    fn batched_render_matches_single_renders() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(40, 32, 0.01);
        // Deliberately different backgrounds: a batch that shared one would not
        // catch per-item background indexing being wrong.
        let scenes: Vec<Scene> = (0..5)
            .map(|i| {
                let mut s = test_scene(6);
                s.background = [i as f32 * 0.15, 0.2, 0.5];
                s
            })
            .collect();

        let batched = gpu.render_many(&scenes, p).expect("batched");
        assert_eq!(batched.len(), scenes.len());
        for (i, scene) in scenes.iter().enumerate() {
            let single = gpu.render(scene, p).expect("single");
            assert_eq!(single.data, batched[i].data, "item {i} differs");
        }
    }

    #[test]
    fn batched_render_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(32, 32, 0.008);
        let scenes: Vec<Scene> = (0..3).map(|_| test_scene(10)).collect();

        for (i, img) in gpu.render_many(&scenes, p).expect("batched").iter().enumerate() {
            let diff = max_abs_diff(&cpu_render(&scenes[i], p).data, &img.data);
            assert!(diff < 1e-5, "item {i}: max diff {diff}");
        }
    }

    #[test]
    fn batched_backward_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(40, 40, 0.02);
        let scenes: Vec<Scene> = (0..4).map(|_| test_scene(8)).collect();
        let targets: Vec<Canvas> =
            (0..4).map(|i| Canvas::filled(40, 40, [0.2 + i as f32 * 0.1, 0.4, 0.6])).collect();

        let batched = gpu.backward_many(&scenes, p, &targets).expect("batched");
        assert_eq!(batched.len(), 4);

        for (i, (loss, grads)) in batched.iter().enumerate() {
            let (rendered, tape) = render_with_tape(&scenes[i], p);
            let (cpu_loss, cpu_grads) = cpu_backward(&scenes[i], p, &tape, &rendered, &targets[i]);

            assert!((cpu_loss - loss).abs() < 1e-6, "item {i} loss {cpu_loss} vs {loss}");
            let norm = cpu_grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
            let err = max_abs_diff(&cpu_grads, grads) / norm;
            assert!(err < 1e-3, "item {i} gradient error {err}");
        }
    }

    #[test]
    fn batch_preserves_order() {
        // Distinct targets per item, so a shuffled result would pair the wrong
        // gradient with the wrong scene and show up as a loss mismatch.
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(24, 24, 0.02);
        let scenes: Vec<Scene> = (0..6).map(|_| test_scene(5)).collect();
        let targets: Vec<Canvas> = (0..6)
            .map(|i| Canvas::filled(24, 24, [i as f32 / 6.0, 0.5, 1.0 - i as f32 / 6.0]))
            .collect();

        let batched = gpu.backward_many(&scenes, p, &targets).expect("batched");
        for (i, (loss, _)) in batched.iter().enumerate() {
            let expected = gpu.render(&scenes[i], p).expect("render").mse(&targets[i]);
            assert!((loss - expected).abs() < 1e-6, "item {i} out of order");
        }
    }

    #[test]
    fn ragged_batches_are_rejected() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(16, 16, 0.02);
        let scenes = vec![test_scene(4), test_scene(7)];
        assert!(gpu.render_many(&scenes, p).is_err());
    }

    #[test]
    fn empty_batch_is_empty_not_an_error() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(16, 16, 0.02);
        assert!(gpu.render_many(&[], p).expect("empty").is_empty());
        assert!(gpu.backward_many(&[], p, &[]).expect("empty").is_empty());
    }

    #[test]
    fn tape_size_is_reported() {
        assert_eq!(GpuRasterizer::tape_bytes(2, 4, 4), 2 * 4 * 4 * 3 * 4);
    }
}
