// Geometry shared by the forward and backward shaders.
//
// These functions mirror `crates/diffrast/src/raster.rs` exactly, including the
// bounding-box cull. That last part matters more than it looks: the CPU culls
// pixels more than six sigma from a triangle, where coverage is still ~2.5e-3 —
// well above the weight threshold. A GPU version that skipped the cull would
// composite contributions the CPU discards and quietly disagree with it.

struct Params {
    width: u32,
    height: u32,
    n_tris: u32,
    sigma: f32,
    background: vec3<f32>,
    min_weight: f32,
    write_tape: u32,
    // Three scalar u32s, not a vec3: a vec3 would align to 16 and push the
    // struct to 64 bytes, silently disagreeing with the 48-byte Rust struct.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// One triangle: 10 floats, matching the CPU parameter layout.
const STRIDE: u32 = 10u;

// Declared here rather than in each entry point: WGSL forbids passing storage
// pointers as function arguments, so the accessors below have to reach the
// buffer at module scope. Both pipelines bind these at the same slots.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tris: array<f32>;

fn tri_vert(i: u32, v: u32) -> vec2<f32> {
    let base = i * STRIDE + v * 2u;
    return vec2<f32>(tris[base], tris[base + 1u]);
}

fn tri_color(i: u32) -> vec3<f32> {
    let base = i * STRIDE + 6u;
    return vec3<f32>(tris[base], tris[base + 1u], tris[base + 2u]);
}

fn tri_alpha(i: u32) -> f32 {
    return tris[i * STRIDE + 9u];
}

fn sigmoid(x: f32) -> f32 {
    // Branch on sign so exp never overflows for large |x|.
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let e = exp(x);
    return e / (1.0 + e);
}

// Distance from `pt` to segment `a -> b`, and the clamped projection parameter.
fn segment_distance(a: vec2<f32>, b: vec2<f32>, pt: vec2<f32>) -> vec2<f32> {
    let ab = b - a;
    let ap = pt - a;
    let len2 = dot(ab, ab);
    var t = 0.0;
    if (len2 > 1e-20) {
        t = clamp(dot(ap, ab) / len2, 0.0, 1.0);
    }
    let d = ap - t * ab;
    return vec2<f32>(t, sqrt(dot(d, d)));
}

fn cross2(a: vec2<f32>, b: vec2<f32>, pt: vec2<f32>) -> f32 {
    return (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
}

// Inside test that tolerates either winding order.
fn is_inside(v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, pt: vec2<f32>) -> bool {
    let c0 = cross2(v0, v1, pt);
    let c1 = cross2(v1, v2, pt);
    let c2 = cross2(v2, v0, pt);
    return (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0) || (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0);
}

// Signed distance to the triangle boundary; positive inside.
// Returns (signed_distance, nearest_edge_index, projection_t).
fn signed_distance(v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, pt: vec2<f32>) -> vec3<f32> {
    let d0 = segment_distance(v0, v1, pt);
    let d1 = segment_distance(v1, v2, pt);
    let d2 = segment_distance(v2, v0, pt);

    var best_edge = 0u;
    var best = d0;
    if (d1.y < best.y) { best_edge = 1u; best = d1; }
    if (d2.y < best.y) { best_edge = 2u; best = d2; }

    var sd = best.y;
    if (!is_inside(v0, v1, v2, pt)) {
        sd = -best.y;
    }
    return vec3<f32>(sd, f32(best_edge), best.x);
}

// Replicates the CPU's integer bounding-box cull, so both paths include exactly
// the same pixels for a given triangle.
fn in_bounds(v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, x: u32, y: u32, p: Params) -> bool {
    let lo = min(min(v0, v1), v2);
    let hi = max(max(v0, v1), v2);
    let pad = 6.0 * p.sigma;

    let w = f32(p.width);
    let h = f32(p.height);
    let x0 = i32(max(floor((lo.x - pad) * w), 0.0));
    let y0 = i32(max(floor((lo.y - pad) * h), 0.0));
    let x1 = min(i32(max(ceil((hi.x + pad) * w), 0.0)) + 1, i32(p.width));
    let y1 = min(i32(max(ceil((hi.y + pad) * h), 0.0)) + 1, i32(p.height));

    let xi = i32(x);
    let yi = i32(y);
    return xi >= x0 && xi < x1 && yi >= y0 && yi < y1;
}

fn pixel_center(x: u32, y: u32, p: Params) -> vec2<f32> {
    return vec2<f32>((f32(x) + 0.5) / f32(p.width), (f32(y) + 0.5) / f32(p.height));
}
