// Batched backward pass, tape-free.
//
// Only the recompute strategy is offered here. A batched tape would need
// `batch x triangles x pixels x 3` floats — 12 GB for a batch of 32 at 256px
// with 512 triangles, which exceeds a 4090's memory for a workload that fits
// comfortably otherwise. Recomputing wins even unbatched (1.49x at 512
// triangles), so there is nothing to trade away.

@group(0) @binding(2) var<storage, read> rendered: array<f32>;
@group(0) @binding(3) var<storage, read> target_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> grads: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> backgrounds: array<f32>;

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

// Canvas state just before triangle `upto` (absolute index) was composited.
fn canvas_before(tri_base: u32, upto: u32, b: u32, pt: vec2<f32>, x: u32, y: u32) -> vec3<f32> {
    let bg = b * 3u;
    var color = vec3<f32>(backgrounds[bg], backgrounds[bg + 1u], backgrounds[bg + 2u]);
    for (var k = 0u; k < upto; k = k + 1u) {
        let j = tri_base + k;
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
    let b = gid.z;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let pt = pixel_center(x, y, params);
    let pixels = params.width * params.height;
    let pixel = y * params.width + x;
    let o = (b * pixels + pixel) * 3u;

    // Normalized per image, not per batch, so each item's loss matches what the
    // single-scene path reports and batching stays a pure scheduling change.
    let n = f32(pixels * 3u);
    let scale = 2.0 / n;
    var g = vec3<f32>(
        scale * (rendered[o] - target_image[o]),
        scale * (rendered[o + 1u] - target_image[o + 1u]),
        scale * (rendered[o + 2u] - target_image[o + 2u]),
    );

    let sigma = max(params.sigma, 1e-8);
    let tri_base = b * params.n_tris;

    for (var k = 0u; k < params.n_tris; k = k + 1u) {
        let idx = params.n_tris - 1u - k;
        let i = tri_base + idx;

        let raw_alpha = tri_alpha(i);
        let alpha = clamp(raw_alpha, 0.0, 1.0);
        if (alpha <= 0.0) {
            continue;
        }

        let v0 = tri_vert(i, 0u);
        let v1 = tri_vert(i, 1u);
        let v2 = tri_vert(i, 2u);
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

        let dst = canvas_before(tri_base, idx, b, pt, x, y);
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
        var bb = v1;
        if (edge == 1u) { a = v1; bb = v2; }
        if (edge == 2u) { a = v2; bb = v0; }

        let cp = a + t * (bb - a);
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
