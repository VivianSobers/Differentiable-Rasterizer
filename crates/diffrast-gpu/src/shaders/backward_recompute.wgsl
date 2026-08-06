// Backward pass without a tape: recompute the canvas state instead of storing it.
//
// The taped variant writes `triangles x pixels x 3` floats in the forward pass
// and reads them all back here — 384 MB for 512 triangles at 256px. On any
// device where bandwidth is scarcer than arithmetic (integrated GPUs, and to a
// lesser degree discrete ones), paying that transfer costs more than the
// compute it saves.
//
// This variant stores nothing. For each triangle it re-walks the triangles
// beneath it to rebuild the canvas state at that point, which is O(T^2) per
// pixel instead of O(T). The trade is a large amount of arithmetic for zero
// memory traffic — and the arithmetic is far cheaper than it sounds, because
// the bounding-box cull rejects most (triangle, pixel) pairs in a few
// instructions.
//
// Which variant wins is a property of the device, not of the algorithm, so both
// are kept and `GpuRasterizer::backward` picks between them.

@group(0) @binding(2) var<storage, read> rendered: array<f32>;
@group(0) @binding(3) var<storage, read> target_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> grads: array<atomic<u32>>;

fn atomic_add_f32(index: u32, value: f32) {
    if (value == 0.0) {
        return;
    }
    var old = atomicLoad(&grads[index]);
    loop {
        let updated = bitcast<u32>(bitcast<f32>(old) + value);
        let result = atomicCompareExchangeWeak(&grads[index], old, updated);
        if (result.exchanged) {
            break;
        }
        old = result.old_value;
    }
}

// Coverage weight of triangle `i` at `pt`, or 0 if it does not contribute.
// Mirrors the forward shader's skip conditions exactly.
fn weight_at(i: u32, pt: vec2<f32>, x: u32, y: u32) -> f32 {
    let alpha = clamp(tri_alpha(i), 0.0, 1.0);
    if (alpha <= 0.0) {
        return 0.0;
    }
    let v0 = tri_vert(i, 0u);
    let v1 = tri_vert(i, 1u);
    let v2 = tri_vert(i, 2u);
    if (!in_bounds(v0, v1, v2, x, y, params)) {
        return 0.0;
    }
    let sd = signed_distance(v0, v1, v2, pt).x;
    let w = alpha * sigmoid(sd / max(params.sigma, 1e-8));
    if (w <= params.min_weight) {
        return 0.0;
    }
    return w;
}

// Canvas state immediately before triangle `upto` was composited.
fn canvas_before(upto: u32, pt: vec2<f32>, x: u32, y: u32) -> vec3<f32> {
    var color = params.background;
    for (var j = 0u; j < upto; j = j + 1u) {
        let w = weight_at(j, pt, x, y);
        if (w == 0.0) {
            continue;
        }
        color = tri_color(j) * w + color * (1.0 - w);
    }
    return color;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let pt = pixel_center(x, y, params);
    let pixel = y * params.width + x;
    let o = pixel * 3u;

    let n = f32(params.width * params.height * 3u);
    let scale = 2.0 / n;
    var g = vec3<f32>(
        scale * (rendered[o] - target_image[o]),
        scale * (rendered[o + 1u] - target_image[o + 1u]),
        scale * (rendered[o + 2u] - target_image[o + 2u]),
    );

    let sigma = max(params.sigma, 1e-8);

    for (var k = 0u; k < params.n_tris; k = k + 1u) {
        let i = params.n_tris - 1u - k;

        let v0 = tri_vert(i, 0u);
        let v1 = tri_vert(i, 1u);
        let v2 = tri_vert(i, 2u);

        let raw_alpha = tri_alpha(i);
        let alpha = clamp(raw_alpha, 0.0, 1.0);
        if (alpha <= 0.0) {
            continue;
        }
        if (!in_bounds(v0, v1, v2, x, y, params)) {
            continue;
        }

        let sdt = signed_distance(v0, v1, v2, pt);
        let sd = sdt.x;
        let cov = sigmoid(sd / sigma);
        let w = alpha * cov;
        if (w <= params.min_weight) {
            continue;
        }

        // The one extra cost of this variant, and only for triangles that
        // actually contribute at this pixel.
        let dst = canvas_before(i, pt, x, y);
        let color = tri_color(i);
        let base = i * STRIDE;

        atomic_add_f32(base + 6u, g.r * w);
        atomic_add_f32(base + 7u, g.g * w);
        atomic_add_f32(base + 8u, g.b * w);

        let d_w = dot(g, color - dst);
        g = g * (1.0 - w);

        if (raw_alpha > 0.0 && raw_alpha < 1.0) {
            atomic_add_f32(base + 9u, d_w * cov);
        }

        let d_cov = d_w * alpha;
        let dist = abs(sd);
        if (dist < 1e-9) {
            continue;
        }

        let edge = u32(sdt.y);
        let t = sdt.z;

        var a = v0;
        var b = v1;
        if (edge == 1u) { a = v1; b = v2; }
        if (edge == 2u) { a = v2; b = v0; }

        let cp = a + t * (b - a);
        let u = (pt - cp) / dist;

        var sign_sd = -1.0;
        if (sd > 0.0) { sign_sd = 1.0; }
        let chain = d_cov * cov * (1.0 - cov) / sigma * sign_sd;

        let ga = chain * -(1.0 - t) * u;
        let gb = chain * -t * u;

        let ia = edge;
        let ib = (edge + 1u) % 3u;
        atomic_add_f32(base + ia * 2u, ga.x);
        atomic_add_f32(base + ia * 2u + 1u, ga.y);
        atomic_add_f32(base + ib * 2u, gb.x);
        atomic_add_f32(base + ib * 2u + 1u, gb.y);
    }
}
