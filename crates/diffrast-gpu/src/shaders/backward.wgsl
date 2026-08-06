// Backward pass: one thread per pixel, walking the triangle list in reverse.
//
// Two things make this different from the forward pass.
//
// First, the reverse sweep needs the canvas as it stood before each triangle
// was composited. That is what the tape holds — written by the forward shader,
// indexed by (triangle, pixel). The CPU stores only each triangle's bounding
// rectangle; the GPU stores the full canvas per triangle, trading memory for a
// flat indexing scheme that needs no per-triangle offset table.
//
// Second, gradients accumulate *per triangle* while threads are *per pixel*, so
// thousands of threads add into the same ten floats. WGSL has no native f32
// atomic add, so this uses the standard compare-exchange loop on the bit
// pattern. It is exact — no fixed-point quantization — at the cost of retrying
// under contention. Contention is the main thing a tiled reduction would fix.

@group(0) @binding(2) var<storage, read> rendered: array<f32>;
@group(0) @binding(3) var<storage, read> target_image: array<f32>;
@group(0) @binding(4) var<storage, read> tape: array<f32>;
@group(0) @binding(5) var<storage, read_write> grads: array<atomic<u32>>;

// Exact f32 atomic add, built from a u32 compare-exchange.
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
        // Another thread won the race; retry against the value it left.
        old = result.old_value;
    }
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

    // d(mse)/d(output pixel).
    let n = f32(params.width * params.height * 3u);
    let scale = 2.0 / n;
    var g = vec3<f32>(
        scale * (rendered[o] - target_image[o]),
        scale * (rendered[o + 1u] - target_image[o + 1u]),
        scale * (rendered[o + 2u] - target_image[o + 2u]),
    );

    let sigma = max(params.sigma, 1e-8);

    // Front-to-back: the reverse of the compositing order.
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

        let slot = (i * params.width * params.height + pixel) * 3u;
        let dst = vec3<f32>(tape[slot], tape[slot + 1u], tape[slot + 2u]);
        let color = tri_color(i);

        let base = i * STRIDE;

        // out = color * w + dst * (1 - w)
        atomic_add_f32(base + 6u, g.r * w);
        atomic_add_f32(base + 7u, g.g * w);
        atomic_add_f32(base + 8u, g.b * w);

        let d_w = dot(g, color - dst);

        // What reaches the layers underneath, attenuated by this one.
        g = g * (1.0 - w);

        // A clamped alpha is locally constant, so no gradient flows to it.
        if (raw_alpha > 0.0 && raw_alpha < 1.0) {
            atomic_add_f32(base + 9u, d_w * cov);
        }

        let d_cov = d_w * alpha;
        let dist = abs(sd);
        if (dist < 1e-9) {
            // Exactly on the boundary the distance has no direction. Zero is
            // the honest contribution; surrounding pixels supply a valid
            // subgradient.
            continue;
        }

        // d(dist)/da = -(1 - t) * u, d(dist)/db = -t * u, where u points from
        // the closest point on the nearest edge toward this pixel. Holds both
        // when the projection lands inside the segment and when it is clamped
        // to an endpoint.
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
