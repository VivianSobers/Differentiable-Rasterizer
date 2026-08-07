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
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Mutex;

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
    grad_input: u32,
    _pad: [u32; 2],
}

/// Something went wrong talking to the GPU.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    /// The device did not finish the work within [`GpuRasterizer::POLL_TIMEOUT`].
    Timeout,
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
            Self::Timeout => write!(
                f,
                "the GPU did not finish within {:?} — it may be busy with another process \
                 (check `nvidia-smi`), or the workload may be too large for one submission",
                GpuRasterizer::POLL_TIMEOUT
            ),
            Self::DeviceRequest(m) => write!(f, "could not create device: {m}"),
            Self::Readback(m) => write!(f, "could not read results back: {m}"),
            Self::TooLarge(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// How the batched backward pass accumulates gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReduceMode {
    /// One global atomic per contributing pixel.
    Direct,
    /// Accumulate in workgroup memory, then one global atomic per triangle per
    /// workgroup. Default: contention on the accumulators is the measured
    /// bottleneck, and this is what addresses it.
    #[default]
    Workgroup,
}

/// Per-item `(loss, gradients)`, one entry per scene in a batch.
pub type BatchGradients = Vec<(f32, Vec<f32>)>;

/// Where the time went inside one [`GpuRasterizer::backward_many`] call.
///
/// Added after two wrong guesses about the bottleneck. Extrapolating the
/// benchmark to zero triangles left ~79 ms per batch at 256px that no amount of
/// geometry explained, and reasoning about which line was responsible produced
/// the wrong answer twice. Measuring each phase is cheaper than another guess.
#[derive(Clone, Copy, Debug, Default)]
pub struct BackwardTimings {
    /// Creating and filling the triangle, background and parameter buffers.
    pub pack_ms: f32,
    /// Allocating the image, target and gradient buffers.
    pub alloc_ms: f32,
    /// Recording both passes and waiting for the gradients — includes all GPU
    /// execution, since reading the gradients forces a full sync.
    pub dispatch_ms: f32,
    /// Copying the rendered images back to the host.
    pub readback_ms: f32,
    /// Reducing the per-item loss on the CPU.
    pub loss_ms: f32,
}

impl BackwardTimings {
    pub fn total_ms(&self) -> f32 {
        self.pack_ms + self.alloc_ms + self.dispatch_ms + self.readback_ms + self.loss_ms
    }
}

/// Buffers for one batch, uploaded once and shared by both dispatches.
struct PackedBatch {
    params: wgpu::Buffer,
    tris: wgpu::Buffer,
    backgrounds: wgpu::Buffer,
}

/// How many buffers the pool will hold on to, in bytes.
///
/// Generous, because the buffers worth pooling are the large ones and a
/// training run reuses the same few sizes forever. Past the cap, released
/// buffers are dropped rather than kept, so a caller that sweeps through many
/// distinct sizes degrades to plain allocation instead of growing without
/// bound.
const POOL_BYTE_LIMIT: u64 = 512 << 20;

/// Reusable device buffers, keyed by `(usage, size)`.
///
/// Allocation became the largest phase of a batched backward call once atomic
/// contention was fixed — 5.85 ms of 13.04 ms at 256px, against 1.95 ms of
/// actual dispatch. Training calls this thousands of times at *identical*
/// shapes, so every allocation after the first is avoidable.
///
/// Sizes must match exactly to be reused. Bucketing by rounding up would raise
/// the hit rate for callers that vary their shapes, but training does not vary
/// them, and exact matching keeps the mapping between a binding's length and
/// its contents obvious.
#[derive(Default)]
struct BufferPool {
    free: HashMap<(u32, u64), Vec<wgpu::Buffer>>,
    bytes: u64,
    hits: u64,
    misses: u64,
}

/// Pool counters, exposed so reuse can be *asserted* rather than inferred from
/// a timing. A pool that silently never hits looks exactly like a slow GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes: u64,
}

/// A buffer borrowed from the pool, returned when it goes out of scope.
///
/// Every call site reads its results back before returning, which blocks until
/// the queue drains, so a buffer is always idle by the time it is released.
struct Pooled<'a> {
    buffer: Option<wgpu::Buffer>,
    pool: &'a Mutex<BufferPool>,
    key: (u32, u64),
}

impl Deref for Pooled<'_> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("buffer is taken only on drop")
    }
}

impl Drop for Pooled<'_> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else { return };
        let Ok(mut pool) = self.pool.lock() else { return };
        if pool.bytes + self.key.1 > POOL_BYTE_LIMIT {
            return;
        }
        pool.bytes += self.key.1;
        pool.free.entry(self.key).or_default().push(buffer);
    }
}

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
    backward_batch_reduced: wgpu::ComputePipeline,
    loss_batch: wgpu::ComputePipeline,
    pool: Mutex<BufferPool>,
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
    /// How long to wait for a submission before giving up.
    ///
    /// Bounded on purpose. An unbounded wait turns any stall — a device shared
    /// with another process, a lost submission, a driver hiccup — into a
    /// process that hangs forever with no diagnostic. Failing with a message
    /// after a minute is strictly better than blocking indefinitely.
    pub const POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

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

        let backward_batch_reduced = Self::pipeline(
            &device,
            "backward_batch_reduced",
            &format!("{common}\n{}", include_str!("shaders/backward_batch_reduced.wgsl")),
        );

        // No `common` prefix: this one touches no geometry.
        let loss_batch =
            Self::pipeline(&device, "loss_batch", include_str!("shaders/loss_batch.wgsl"));

        Ok(Self {
            device,
            queue,
            forward,
            backward,
            backward_recompute,
            forward_batch,
            backward_batch,
            backward_batch_reduced,
            loss_batch,
            pool: Mutex::new(BufferPool::default()),
            info,
        })
    }

    /// Take a buffer of exactly this size and usage from the pool, allocating
    /// one only if none is free. The contents are whatever the last user left,
    /// so callers must overwrite or clear it.
    fn pooled(&self, label: &'static str, usage: wgpu::BufferUsages, size: u64) -> Pooled<'_> {
        let key = (usage.bits(), size);
        let mut pool = self.pool.lock().expect("pool mutex poisoned");

        if let Some(buffer) = pool.free.get_mut(&key).and_then(Vec::pop) {
            pool.bytes -= size;
            pool.hits += 1;
            return Pooled { buffer: Some(buffer), pool: &self.pool, key };
        }

        pool.misses += 1;
        drop(pool);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Pooled { buffer: Some(buffer), pool: &self.pool, key }
    }

    /// Reuse counters for the buffer pool.
    pub fn pool_stats(&self) -> PoolStats {
        let pool = self.pool.lock().expect("pool mutex poisoned");
        PoolStats { hits: pool.hits, misses: pool.misses, bytes: pool.bytes }
    }

    /// Drop every pooled buffer, returning the memory to the driver.
    pub fn clear_pool(&self) {
        let mut pool = self.pool.lock().expect("pool mutex poisoned");
        pool.free.clear();
        pool.bytes = 0;
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
            grad_input: 0,
            _pad: [0; 2],
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

        let packed = self.pack_batch(scenes, p, n_tris);
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let image_buf = self.image_buffer(p, batch);
        self.encode_forward(&mut encoder, &packed, p, batch, &image_buf);

        let data = self.read_buffer(&mut encoder, &image_buf, (batch * pixels * 3 * 4) as u64)?;
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
    ) -> Result<BatchGradients, GpuError> {
        self.backward_many_inner(scenes, p, targets, ReduceMode::default(), false).map(|(r, _)| r)
    }

    /// As [`Self::backward_many`], but also reports where the time went.
    ///
    /// **Measuring costs something.** Phases are separated by waiting for the
    /// queue to drain between them; without that the whole call is recorded
    /// into one encoder and submitted once, and every phase before the
    /// readback measures nothing but CPU-side recording. So this is slightly
    /// slower than [`Self::backward_many`] and its total is an upper bound.
    /// The breakdown is a diagnostic, not the number to quote.
    pub fn backward_many_timed(
        &self,
        scenes: &[Scene],
        p: RenderParams,
        targets: &[Canvas],
    ) -> Result<(BatchGradients, BackwardTimings), GpuError> {
        self.backward_many_inner(scenes, p, targets, ReduceMode::default(), true)
    }

    /// Full form: choose how gradients are reduced, and get phase timings back.
    pub fn backward_many_full(
        &self,
        scenes: &[Scene],
        p: RenderParams,
        targets: &[Canvas],
        reduce: ReduceMode,
    ) -> Result<(BatchGradients, BackwardTimings), GpuError> {
        self.backward_many_inner(scenes, p, targets, reduce, false)
    }

    fn backward_many_inner(
        &self,
        scenes: &[Scene],
        p: RenderParams,
        targets: &[Canvas],
        reduce: ReduceMode,
        sync_phases: bool,
    ) -> Result<(BatchGradients, BackwardTimings), GpuError> {
        let mut t = BackwardTimings::default();
        let mut clock = std::time::Instant::now();
        let mut lap = |t: &mut f32| {
            *t = clock.elapsed().as_secs_f32() * 1e3;
            clock = std::time::Instant::now();
        };

        if scenes.len() != targets.len() {
            return Err(GpuError::TooLarge(format!(
                "batch mismatch: {} scenes but {} targets",
                scenes.len(),
                targets.len()
            )));
        }
        let Some(n_tris) = self.check_batch(scenes)? else {
            return Ok((Vec::new(), t));
        };

        let batch = scenes.len();
        let pixels = p.width * p.height;
        let image_bytes = (batch * pixels * 3 * 4) as u64;

        // Checked rather than assumed: the target buffer is now sized from the
        // render parameters, so a target of the wrong size would write past its
        // end. Previously it produced a mis-sized binding and quietly wrong
        // gradients, which is worse.
        if let Some(bad) = targets.iter().position(|c| c.width != p.width || c.height != p.height) {
            return Err(GpuError::TooLarge(format!(
                "target {bad} is {}x{}, but the render is {}x{}",
                targets[bad].width, targets[bad].height, p.width, p.height
            )));
        }

        let packed = self.pack_batch(scenes, p, n_tris);
        lap(&mut t.pack_ms);

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // The rendered batch stays in GPU memory and feeds the backward pass
        // directly. An earlier version rendered via `render_many`, which copied
        // the images to the host and immediately uploaded them back — 12.6 MB
        // round-tripped per call at 256px, which showed up as GPU cost scaling
        // *superlinearly* with pixel count while scaling sublinearly with
        // triangle count. Compute does not behave that way; transfers do.
        let image_buf = self.image_buffer(p, batch);
        self.encode_forward(&mut encoder, &packed, p, batch, &image_buf);

        // Written per canvas rather than through a concatenated `Vec`, which
        // was a second host-side copy of the whole batch — 12.6 MB at 256px,
        // allocated and memcpy'd once per call for no reason.
        let target_buf = self.pooled(
            "targets",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            image_bytes,
        );
        for (i, target) in targets.iter().enumerate() {
            let offset = (i * pixels * 3 * 4) as u64;
            self.queue.write_buffer(&target_buf, offset, bytemuck::cast_slice(&target.data));
        }

        let grad_len = batch * n_tris * Triangle::N_PARAMS;
        let grad_buf = self.gradient_buffer(grad_len, &mut encoder);
        lap(&mut t.alloc_ms);

        let pipeline = match reduce {
            ReduceMode::Direct => &self.backward_batch,
            ReduceMode::Workgroup => &self.backward_batch_reduced,
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backward_batch"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: packed.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: packed.tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: image_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: target_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: packed.backgrounds.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                p.width.div_ceil(8) as u32,
                p.height.div_ceil(8) as u32,
                batch as u32,
            );
        }

        // The loss is reduced on the device, so nothing per-pixel crosses the
        // bus. Reading the batch back to sum it on the host cost more than the
        // backward dispatch itself once contention was fixed.
        let counts_buf = self.loss_counts(pixels);
        let loss_buf = self.pooled(
            "losses",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            (batch * 4) as u64,
        );
        self.encode_loss(&mut encoder, &counts_buf, &image_buf, &target_buf, &loss_buf, batch);

        // Everything above only *records* commands. Nothing has reached the
        // device until a submit, so attributing GPU execution to a phase means
        // draining the queue here — see the note on `backward_many_timed`.
        if sync_phases {
            self.submit_and_wait(&mut encoder)?;
        }
        lap(&mut t.dispatch_ms);

        // One submission and one map for both results instead of two.
        let read = self.read_buffers(
            &mut encoder,
            &[(&grad_buf, (grad_len * 4) as u64), (&loss_buf, (batch * 4) as u64)],
        )?;
        let (grads, losses) = read.split_at(grad_len);
        lap(&mut t.readback_ms);

        let stride = n_tris * Triangle::N_PARAMS;
        let out: BatchGradients =
            losses.iter().zip(grads.chunks_exact(stride)).map(|(l, g)| (*l, g.to_vec())).collect();
        lap(&mut t.loss_ms);

        Ok((out, t))
    }

    /// Gradients from an upstream image gradient rather than a target.
    ///
    /// This is the shape autograd needs: PyTorch hands the layer `dL/d(pixel)`
    /// and expects `dL/d(parameter)` back. Deriving an equivalent target from
    /// the gradient would work, but only after rendering once to invert
    /// `d(mse)/dr = 2(r - t)/N` — so the shader reads the gradient directly
    /// instead, and the batch is rendered exactly once.
    ///
    /// `grad_images` is `(batch, height, width, 3)` flattened, matching the
    /// layout `render_many` produces.
    pub fn backward_many_from_grad(
        &self,
        scenes: &[Scene],
        p: RenderParams,
        grad_images: &[f32],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        let Some(n_tris) = self.check_batch(scenes)? else {
            return Ok(Vec::new());
        };
        let batch = scenes.len();
        let pixels = p.width * p.height;

        let expected = batch * pixels * 3;
        if grad_images.len() != expected {
            return Err(GpuError::TooLarge(format!(
                "expected {expected} gradient values, got {}",
                grad_images.len()
            )));
        }

        let packed = self.pack_batch_with(scenes, p, n_tris, true);
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let image_buf = self.image_buffer(p, batch);
        self.encode_forward(&mut encoder, &packed, p, batch, &image_buf);

        let grad_in_buf = self.pooled(
            "grad_in",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            std::mem::size_of_val(grad_images) as u64,
        );
        self.queue.write_buffer(&grad_in_buf, 0, bytemuck::cast_slice(grad_images));

        let grad_len = batch * n_tris * Triangle::N_PARAMS;
        let grad_buf = self.gradient_buffer(grad_len, &mut encoder);

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backward_batch_grad"),
            layout: &self.backward_batch_reduced.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: packed.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: packed.tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: image_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: packed.backgrounds.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_batch_grad"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backward_batch_reduced);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                p.width.div_ceil(8) as u32,
                p.height.div_ceil(8) as u32,
                batch as u32,
            );
        }

        let grads = self.read_buffer(&mut encoder, &grad_buf, (grad_len * 4) as u64)?;
        let stride = n_tris * Triangle::N_PARAMS;
        Ok(grads.chunks_exact(stride).map(|g| g.to_vec()).collect())
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

    /// Per-batch buffers, uploaded once and reused by both dispatches.
    fn pack_batch(&self, scenes: &[Scene], p: RenderParams, n_tris: usize) -> PackedBatch {
        self.pack_batch_with(scenes, p, n_tris, false)
    }

    fn pack_batch_with(
        &self,
        scenes: &[Scene],
        p: RenderParams,
        n_tris: usize,
        grad_input: bool,
    ) -> PackedBatch {
        let tris: Vec<f32> = scenes.iter().flat_map(|s| s.params()).collect();
        let backgrounds: Vec<f32> = scenes.iter().flat_map(|s| s.background).collect();
        let mut params = Self::gpu_params(&scenes[0], p, false);
        params.n_tris = n_tris as u32;
        params.grad_input = u32::from(grad_input);

        PackedBatch {
            params: self.uniform(&params),
            tris: self.storage("triangles", &tris),
            backgrounds: self.storage("backgrounds", &backgrounds),
        }
    }

    /// Take a pooled buffer sized for one batch of rendered images.
    ///
    /// The forward shader writes every pixel it is dispatched for, so a reused
    /// buffer needs no clearing — which is the point of pooling it.
    fn image_buffer(&self, p: RenderParams, batch: usize) -> Pooled<'_> {
        self.pooled(
            "images",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            (batch * p.width * p.height * 3 * 4) as u64,
        )
    }

    /// Take a pooled gradient accumulator and record a clear for it.
    ///
    /// Unlike the image buffer this one is accumulated into, not overwritten,
    /// so a reused buffer must be zeroed. `clear_buffer` does it on the device;
    /// uploading a zero-filled `Vec` (what allocating it fresh amounted to)
    /// would put the cost straight back.
    fn gradient_buffer(&self, len: usize, encoder: &mut wgpu::CommandEncoder) -> Pooled<'_> {
        let buffer = self.pooled(
            "grads",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            (len * 4) as u64,
        );
        encoder.clear_buffer(&buffer, 0, None);
        buffer
    }

    /// The one uniform the loss shader needs: values per image.
    fn loss_counts(&self, pixels: usize) -> Pooled<'_> {
        let buffer = self.pooled(
            "loss_counts",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            16,
        );
        let counts = [(pixels * 3) as u32, 0, 0, 0];
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&counts));
        buffer
    }

    /// Record the per-item MSE reduction. One workgroup per batch item.
    ///
    /// Every buffer is passed in rather than created here so the caller owns
    /// them for the whole submission. A pooled buffer released before its work
    /// is submitted could be handed to another thread and rewritten underneath
    /// a queued dispatch.
    fn encode_loss(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        counts_buf: &wgpu::Buffer,
        image_buf: &wgpu::Buffer,
        target_buf: &wgpu::Buffer,
        loss_buf: &wgpu::Buffer,
        batch: usize,
    ) {
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loss_batch"),
            layout: &self.loss_batch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: image_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: target_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: loss_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("loss_batch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.loss_batch);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(batch as u32, 1, 1);
    }

    /// Record the batched forward pass into `image_buf`. Nothing is read back
    /// here; the buffer stays on the device for the backward pass to consume.
    fn encode_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        packed: &PackedBatch,
        p: RenderParams,
        batch: usize,
        image_buf: &wgpu::Buffer,
    ) {
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward_batch"),
            layout: &self.forward_batch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: packed.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: packed.tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: image_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: packed.backgrounds.as_entire_binding(),
                },
            ],
        });

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

    /// Submit whatever is recorded and block until the device has finished it.
    fn submit_and_wait(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), GpuError> {
        let recorded = std::mem::replace(
            encoder,
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
        );
        self.queue.submit(Some(recorded.finish()));
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Self::POLL_TIMEOUT),
        }) {
            Ok(_) => Ok(()),
            Err(wgpu::PollError::Timeout) => Err(GpuError::Timeout),
            Err(e) => Err(GpuError::Readback(format!("{e:?}"))),
        }
    }

    /// Copy a storage buffer back to the host as `f32`s.
    fn read_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        size: u64,
    ) -> Result<Vec<f32>, GpuError> {
        self.read_buffers(encoder, &[(source, size)])
    }

    /// Copy several buffers back in one submission, concatenated in order.
    ///
    /// A round trip costs a submit, a map and a poll regardless of how little
    /// it carries, so two small results are worth fetching together.
    fn read_buffers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sources: &[(&wgpu::Buffer, u64)],
    ) -> Result<Vec<f32>, GpuError> {
        let size: u64 = sources.iter().map(|(_, n)| n).sum();
        let staging = self.pooled(
            "staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size,
        );

        let mut encoder = std::mem::replace(
            encoder,
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
        );
        let mut offset = 0;
        for (source, bytes) in sources {
            encoder.copy_buffer_to_buffer(source, 0, &staging, offset, *bytes);
            offset += bytes;
        }
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Self::POLL_TIMEOUT),
        }) {
            Ok(_) => {}
            Err(wgpu::PollError::Timeout) => return Err(GpuError::Timeout),
            Err(e) => return Err(GpuError::Readback(format!("{e:?}"))),
        }

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

    /// One device shared by every test in this binary.
    ///
    /// `cargo test` runs tests on parallel threads, and creating a device per
    /// test meant a dozen simultaneous Vulkan devices on one card. That is
    /// wasteful everywhere and, on a GPU shared with other work, a good way to
    /// exhaust it. Initializing once also skips a dozen shader compilations.
    ///
    /// Returns `None` when no adapter exists, so the suite still runs on
    /// machines without a usable GPU driver.
    fn gpu() -> Option<&'static GpuRasterizer> {
        static GPU: std::sync::OnceLock<Option<GpuRasterizer>> = std::sync::OnceLock::new();
        GPU.get_or_init(|| match GpuRasterizer::new() {
            Ok(g) => {
                eprintln!("gpu: {}", g.adapter_info());
                Some(g)
            }
            Err(e) => {
                eprintln!("skipping GPU tests: {e}");
                None
            }
        })
        .as_ref()
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
    fn device_side_loss_matches_a_host_reduction() {
        // Deliberately large. The host sums in f64 and the shader sums a tree
        // of f32 partials, so the two only have to agree to f32's precision —
        // and the gap grows with how many values are summed. 192x192x3 is
        // 110,592 per image, well past where a naive f32 sum would drift.
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(192, 192, 0.01);
        let scenes: Vec<Scene> = (0..3).map(|_| test_scene(12)).collect();
        let targets: Vec<Canvas> =
            (0..3).map(|i| Canvas::filled(192, 192, [0.9 - i as f32 * 0.3, 0.2, 0.7])).collect();

        let batched = gpu.backward_many(&scenes, p, &targets).expect("batched");
        let rendered = gpu.render_many(&scenes, p).expect("rendered");

        for (i, (loss, _)) in batched.iter().enumerate() {
            let host = rendered[i].mse(&targets[i]);
            assert!((host - loss).abs() < 1e-6, "item {i}: host {host} vs device {loss}");
        }
    }

    #[test]
    fn mismatched_target_sizes_are_rejected() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(16, 16, 0.02);
        let scenes = vec![test_scene(4)];
        let targets = vec![Canvas::filled(16, 12, [0.5, 0.5, 0.5])];
        assert!(gpu.backward_many(&scenes, p, &targets).is_err());
    }

    #[test]
    fn buffers_are_reused_across_identical_calls() {
        // This one builds its own device, unlike every other test here. Pool
        // counters are per-device state and the shared device is in use by
        // tests running concurrently, so exact assertions need isolation.
        let Ok(gpu) = GpuRasterizer::new() else { return };
        let p = RenderParams::new(32, 32, 0.02);
        let scenes: Vec<Scene> = (0..2).map(|_| test_scene(6)).collect();
        let targets: Vec<Canvas> =
            (0..2).map(|_| Canvas::filled(32, 32, [0.3, 0.4, 0.5])).collect();

        let first = gpu.backward_many(&scenes, p, &targets).expect("first call");
        let after_first = gpu.pool_stats();
        assert!(after_first.misses > 0, "the first call must allocate something");

        let second = gpu.backward_many(&scenes, p, &targets).expect("second call");
        let after_second = gpu.pool_stats();

        // The whole claim: a repeat of the same shape allocates nothing. A
        // pool that silently never hit would be invisible in a timing.
        assert_eq!(
            after_second.misses,
            after_first.misses,
            "identical second call allocated {} new buffers",
            after_second.misses - after_first.misses
        );
        assert!(after_second.hits > after_first.hits, "no buffer was reused");

        // Reuse must not leak state between calls — the gradient accumulator
        // is the one that is added into rather than overwritten, so a missing
        // clear would double it here.
        for (a, b) in first.iter().zip(&second) {
            assert!((a.0 - b.0).abs() < 1e-9, "loss changed on reuse: {} vs {}", a.0, b.0);
            assert_eq!(a.1, b.1, "gradients changed when buffers were reused");
        }

        gpu.clear_pool();
        assert_eq!(gpu.pool_stats().bytes, 0);
    }

    #[test]
    fn reduce_modes_agree() {
        // The two paths accumulate gradients by completely different means —
        // one global atomic per pixel versus a workgroup reduction with
        // barriers. Agreement is the check that the barrier restructuring did
        // not drop or double-count a contribution.
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(40, 40, 0.02);
        let scenes: Vec<Scene> = (0..4).map(|_| test_scene(10)).collect();
        let targets: Vec<Canvas> =
            (0..4).map(|i| Canvas::filled(40, 40, [0.2 + i as f32 * 0.1, 0.4, 0.6])).collect();

        let (direct, _) =
            gpu.backward_many_full(&scenes, p, &targets, ReduceMode::Direct).expect("direct");
        let (reduced, _) =
            gpu.backward_many_full(&scenes, p, &targets, ReduceMode::Workgroup).expect("workgroup");

        for (i, ((la, ga), (lb, gb))) in direct.iter().zip(&reduced).enumerate() {
            assert!((la - lb).abs() < 1e-9, "item {i} loss {la} vs {lb}");
            let norm = ga.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
            let err = max_abs_diff(ga, gb) / norm;
            assert!(err < 1e-3, "item {i} gradient error {err}");
        }
    }

    #[test]
    fn workgroup_reduction_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(48, 48, 0.02);
        let scenes: Vec<Scene> = (0..3).map(|_| test_scene(12)).collect();
        let targets: Vec<Canvas> =
            (0..3).map(|_| Canvas::filled(48, 48, [0.5, 0.3, 0.7])).collect();

        let (out, _) =
            gpu.backward_many_full(&scenes, p, &targets, ReduceMode::Workgroup).expect("workgroup");

        for (i, (loss, grads)) in out.iter().enumerate() {
            let (rendered, tape) = render_with_tape(&scenes[i], p);
            let (cpu_loss, cpu_grads) = cpu_backward(&scenes[i], p, &tape, &rendered, &targets[i]);
            assert!((cpu_loss - loss).abs() < 1e-6, "item {i} loss");
            let norm = cpu_grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
            assert!(max_abs_diff(&cpu_grads, grads) / norm < 1e-3, "item {i} gradients");
        }
    }

    #[test]
    fn workgroup_reduction_handles_non_multiple_sizes() {
        // Exercises the awkward cases: a canvas whose dimensions are not a
        // multiple of the 8x8 workgroup (so edge threads are inactive but must
        // still reach every barrier), and a triangle count that is not a
        // multiple of the 32-triangle chunk.
        let Some(gpu) = gpu() else { return };
        for (w, h, tris) in [(37usize, 19usize, 33usize), (8, 8, 1), (17, 41, 65)] {
            let p = RenderParams::new(w, h, 0.02);
            let scenes = vec![test_scene(tris)];
            let targets = vec![Canvas::filled(w, h, [0.3, 0.5, 0.7])];

            let (out, _) = gpu
                .backward_many_full(&scenes, p, &targets, ReduceMode::Workgroup)
                .expect("workgroup");

            let (rendered, tape) = render_with_tape(&scenes[0], p);
            let (_, cpu_grads) = cpu_backward(&scenes[0], p, &tape, &rendered, &targets[0]);
            let norm = cpu_grads.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
            let err = max_abs_diff(&cpu_grads, &out[0].1) / norm;
            assert!(err < 1e-3, "{w}x{h} with {tris} triangles: error {err}");
        }
    }

    #[test]
    fn gradient_input_mode_matches_target_mode() {
        // Feeding the shader d(mse)/d(pixel) directly must reproduce what it
        // computes internally from a target. This is the path autograd uses,
        // so a discrepancy would corrupt every trained model while leaving the
        // standalone fitter perfectly correct.
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(32, 32, 0.02);
        let scenes: Vec<Scene> = (0..3).map(|_| test_scene(8)).collect();
        let targets: Vec<Canvas> =
            (0..3).map(|i| Canvas::filled(32, 32, [0.2 * i as f32, 0.5, 0.7])).collect();

        let via_target = gpu.backward_many(&scenes, p, &targets).expect("target mode");

        // Build the equivalent upstream gradient: d(mse)/dr = 2(r - t)/N.
        let rendered = gpu.render_many(&scenes, p).expect("render");
        let n = (32 * 32 * 3) as f32;
        let grad_in: Vec<f32> = rendered
            .iter()
            .zip(&targets)
            .flat_map(|(r, t)| {
                r.data.iter().zip(&t.data).map(|(a, b)| 2.0 * (a - b) / n).collect::<Vec<_>>()
            })
            .collect();

        let via_grad = gpu.backward_many_from_grad(&scenes, p, &grad_in).expect("grad mode");

        for (i, ((_, a), b)) in via_target.iter().zip(&via_grad).enumerate() {
            let norm = a.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-12);
            let err = max_abs_diff(a, b) / norm;
            assert!(err < 1e-3, "item {i}: relative difference {err}");
        }
    }

    #[test]
    fn gradient_input_rejects_wrong_length() {
        let Some(gpu) = gpu() else { return };
        let p = RenderParams::new(16, 16, 0.02);
        let scenes = vec![test_scene(4)];
        assert!(gpu.backward_many_from_grad(&scenes, p, &[0.0; 10]).is_err());
    }

    #[test]
    fn tape_size_is_reported() {
        assert_eq!(GpuRasterizer::tape_bytes(2, 4, 4), 2 * 4 * 4 * 3 * 4);
    }
}
