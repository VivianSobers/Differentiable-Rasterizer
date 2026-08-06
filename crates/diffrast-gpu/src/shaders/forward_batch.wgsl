// Batched forward pass: one dispatch for a whole batch of scenes.
//
// Motivated by measurement rather than principle. On a 4090, going from 128 to
// 512 triangles at fixed resolution costs 1.08x the time while the CPU pays
// 3.87x — the device is nowhere near saturated, and what a single small render
// actually costs is the fixed price of a dispatch plus the upload and readback
// around it. Submitting a batch as one dispatch pays that price once instead of
// once per item.
//
// The scene index rides in `gid.z`. Triangles are addressed absolutely, as
// `batch * n_tris + i`, which means every accessor in common.wgsl works
// unchanged — the batch dimension costs no extra indexing machinery.
//
// All scenes in a batch must share a triangle count, which training satisfies
// by construction.

@group(0) @binding(2) var<storage, read_write> out_image: array<f32>;
// Per-scene clear color, three floats each. A separate buffer rather than a
// uniform field because batch items legitimately differ here.
@group(0) @binding(3) var<storage, read> backgrounds: array<f32>;

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

    let bg = b * 3u;
    var color = vec3<f32>(backgrounds[bg], backgrounds[bg + 1u], backgrounds[bg + 2u]);

    let tri_base = b * params.n_tris;

    for (var k = 0u; k < params.n_tris; k = k + 1u) {
        let i = tri_base + k;

        let alpha = clamp(tri_alpha(i), 0.0, 1.0);
        if (alpha <= 0.0) {
            continue;
        }

        let v0 = tri_vert(i, 0u);
        let v1 = tri_vert(i, 1u);
        let v2 = tri_vert(i, 2u);
        if (!in_bounds(v0, v1, v2, x, y, params)) {
            continue;
        }

        let sd = signed_distance(v0, v1, v2, pt).x;
        let w = alpha * sigmoid(sd / max(params.sigma, 1e-8));
        if (w <= params.min_weight) {
            continue;
        }

        color = tri_color(i) * w + color * (1.0 - w);
    }

    let o = (b * pixels + pixel) * 3u;
    out_image[o] = color.r;
    out_image[o + 1u] = color.g;
    out_image[o + 2u] = color.b;
}
