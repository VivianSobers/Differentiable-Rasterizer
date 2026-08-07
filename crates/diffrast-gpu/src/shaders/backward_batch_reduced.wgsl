// Batched backward pass with a per-workgroup gradient reduction.
//
// The measured bottleneck: 86.6% of a batched backward call is dispatch, and
// per-triangle cost *falls* 7.2x as triangle count rises — 16x the work in 2.2x
// the time. That is not a compute curve. Gradients accumulate into ten floats
// per triangle while threads run per pixel, so thousands of threads queue on a
// handful of addresses through compare-exchange retries.
//
// This version accumulates into workgroup-shared memory first and performs one
// global atomic per triangle per *workgroup* rather than per pixel. With a
// 64-thread workgroup that is up to 64x fewer global atomics, and the remaining
// contention is between 64 threads on fast shared memory instead of thousands
// on global memory.
//
// ## Barrier uniformity
//
// `workgroupBarrier` is undefined behaviour unless every thread in the
// workgroup reaches it. Two things in the straightforward version violate that,
// and both are restructured here:
//
//   * The early `return` for out-of-range pixels becomes an `in_range` predicate,
//     so edge workgroups still reach every barrier.
//   * The per-triangle `continue` stays, but barriers are placed only in the
//     outer chunk loop, whose trip count depends solely on `params.n_tris` —
//     uniform across the workgroup.
//
// Getting this wrong does not produce a wrong number; it produces a hang or a
// silently corrupted result on some hardware and not others.

@group(0) @binding(2) var<storage, read> rendered: array<f32>;
@group(0) @binding(3) var<storage, read> target_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> grads: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> backgrounds: array<f32>;

// Triangles handled per chunk. Bounds shared memory at CHUNK * 10 * 4 bytes;
// 32 gives 1280 bytes, comfortably inside every device's workgroup budget.
const CHUNK: u32 = 32u;
const THREADS: u32 = 64u; // 8 x 8

var<workgroup> partial: array<atomic<u32>, 320>; // CHUNK * 10

// Compare-exchange f32 add against workgroup memory.
fn workgroup_add_f32(index: u32, value: f32) {
    if (value == 0.0) {
        return;
    }
    var old = atomicLoad(&partial[index]);
    loop {
        let updated = bitcast<u32>(bitcast<f32>(old) + value);
        let result = atomicCompareExchangeWeak(&partial[index], old, updated);
        if (result.exchanged) {
            break;
        }
        old = result.old_value;
    }
}

fn global_add_f32(index: u32, value: f32) {
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
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let x = gid.x;
    let y = gid.y;
    let b = gid.z;

    // No early return: out-of-range threads run the whole function so that
    // every barrier below is reached by the entire workgroup.
    let in_range = (x < params.width) && (y < params.height);

    let pixels = params.width * params.height;
    let pt = pixel_center(x, y, params);
    let sigma = max(params.sigma, 1e-8);
    let tri_base = b * params.n_tris;

    var g = vec3<f32>(0.0, 0.0, 0.0);
    if (in_range) {
        let o = (b * pixels + y * params.width + x) * 3u;
        let scale = 2.0 / f32(pixels * 3u);
        g = vec3<f32>(
            scale * (rendered[o] - target_image[o]),
            scale * (rendered[o + 1u] - target_image[o + 1u]),
            scale * (rendered[o + 2u] - target_image[o + 2u]),
        );
    }

    // Chunks run front-to-back, matching the reverse compositing order. The
    // trip count depends only on n_tris, so it is uniform across the workgroup.
    let n_chunks = (params.n_tris + CHUNK - 1u) / CHUNK;

    for (var c = 0u; c < n_chunks; c = c + 1u) {
        let hi = params.n_tris - c * CHUNK;
        var lo = 0u;
        if (hi > CHUNK) {
            lo = hi - CHUNK;
        }

        // Clear this chunk's accumulators, striped across the workgroup.
        for (var k = lid; k < CHUNK * 10u; k = k + THREADS) {
            atomicStore(&partial[k], 0u);
        }
        workgroupBarrier();

        if (in_range) {
            var idx = hi;
            while (idx > lo) {
                idx = idx - 1u;
                let i = tri_base + idx;
                let slot = (idx - lo) * 10u;

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

                workgroup_add_f32(slot + 6u, g.r * w);
                workgroup_add_f32(slot + 7u, g.g * w);
                workgroup_add_f32(slot + 8u, g.b * w);

                let d_w = dot(g, color - dst);
                g = g * (1.0 - w);

                if (raw_alpha > 0.0 && raw_alpha < 1.0) {
                    workgroup_add_f32(slot + 9u, d_w * cov);
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
                workgroup_add_f32(slot + ia * 2u, ga.x);
                workgroup_add_f32(slot + ia * 2u + 1u, ga.y);
                workgroup_add_f32(slot + ib * 2u, gb.x);
                workgroup_add_f32(slot + ib * 2u + 1u, gb.y);
            }
        }
        workgroupBarrier();

        // One global atomic per accumulator per workgroup, instead of one per
        // contributing pixel. This is the whole point of the pass.
        for (var k = lid; k < CHUNK * 10u; k = k + THREADS) {
            let tri_local = lo + k / 10u;
            if (tri_local >= hi) {
                continue;
            }
            let value = bitcast<f32>(atomicLoad(&partial[k]));
            if (value != 0.0) {
                global_add_f32((tri_base + tri_local) * STRIDE + (k % 10u), value);
            }
        }
        // Keeps the next chunk's clear from racing this flush.
        workgroupBarrier();
    }
}
