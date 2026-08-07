// Per-item mean squared error, reduced on the device.
//
// Without this the host reads the whole rendered batch back purely to sum it —
// 12.6 MB at 256px for a batch of 16, to produce 16 floats. Once atomic
// contention was fixed that readback plus its host-side reduction was 40% of a
// batched backward call, against 15% for the dispatch doing the actual work.
//
// Deliberately standalone: it shares no bindings with `common.wgsl`, because
// the geometry bindings that file declares would have to be supplied to a
// shader that never looks at a triangle.

// x = values per item (width * height * 3). Passed rather than derived from
// `arrayLength`, so the shader does not silently depend on the image buffer
// being sized exactly to the batch.
@group(0) @binding(0) var<uniform> counts: vec4<u32>;
@group(0) @binding(1) var<storage, read> rendered: array<f32>;
@group(0) @binding(2) var<storage, read> target_image: array<f32>;
@group(0) @binding(3) var<storage, read_write> losses: array<f32>;

const THREADS: u32 = 256u;

var<workgroup> partial: array<f32, THREADS>;

// One workgroup per batch item: each reduces its own image and nothing is
// shared between them, so there is no global atomic here at all.
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let n = counts.x;
    let base = wid.x * n;

    // Strided so consecutive threads read consecutive addresses.
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= n) {
            break;
        }
        let d = rendered[base + i] - target_image[base + i];
        acc = acc + d * d;
        i = i + THREADS;
    }
    partial[lid] = acc;
    workgroupBarrier();

    // Tree reduction. The loop's trip count depends only on THREADS, and the
    // barrier sits outside the `lid < stride` guard, so every thread reaches
    // every barrier — the uniformity requirement that makes this legal.
    var stride = THREADS / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid < stride) {
            partial[lid] = partial[lid] + partial[lid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid == 0u) {
        losses[wid.x] = partial[0] / f32(n);
    }
}
