// Forward pass: one thread per pixel, looping over triangles.
//
// This is the inversion that makes the GPU port work at all. The CPU renders
// triangle-by-triangle, and each triangle's composite depends on what the
// previous one left in the framebuffer — a dependency that serializes the whole
// loop. Flipping it so each thread owns one *pixel* and walks the triangle list
// itself makes every thread independent: the sequential chain still exists, but
// it now runs inside a thread instead of across them.
//
// Cost is O(pixels x triangles) rather than O(covered area), which a tiled
// binning pass would improve. At the sizes this is used for, the cull below
// already skips the great majority of that work.

@group(0) @binding(2) var<storage, read_write> out_image: array<f32>;
// Per-pixel, per-triangle canvas state before each triangle was composited —
// the GPU equivalent of the CPU tape. Only written when `write_tape` is set.
@group(0) @binding(3) var<storage, read_write> tape: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let pt = pixel_center(x, y, params);
    let pixel = (y * params.width + x);
    var color = params.background;

    for (var i = 0u; i < params.n_tris; i = i + 1u) {
        let v0 = tri_vert(i, 0u);
        let v1 = tri_vert(i, 1u);
        let v2 = tri_vert(i, 2u);

        // Record the canvas as it stands before this triangle, whether or not
        // the triangle ends up contributing — the backward pass indexes the
        // tape by triangle, so entries must line up even for skipped ones.
        if (params.write_tape != 0u) {
            let slot = (i * params.width * params.height + pixel) * 3u;
            tape[slot] = color.r;
            tape[slot + 1u] = color.g;
            tape[slot + 2u] = color.b;
        }

        let alpha = clamp(tri_alpha(i), 0.0, 1.0);
        if (alpha <= 0.0) {
            continue;
        }
        if (!in_bounds(v0, v1, v2, x, y, params)) {
            continue;
        }

        let sd = signed_distance(v0, v1, v2, pt).x;
        let cov = sigmoid(sd / max(params.sigma, 1e-8));
        let w = alpha * cov;
        if (w <= params.min_weight) {
            continue;
        }

        color = tri_color(i) * w + color * (1.0 - w);
    }

    let o = pixel * 3u;
    out_image[o] = color.r;
    out_image[o + 1u] = color.g;
    out_image[o + 2u] = color.b;
}
