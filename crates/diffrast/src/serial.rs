//! JSON export for fitted scenes.
//!
//! Written by hand rather than with `serde`, because the schema is four fields
//! deep and a serialization dependency would be carried into the WebAssembly
//! build for no benefit. The output is what the Python tools and the web viewer
//! consume.

use crate::scene::Scene;

/// Serialize a scene as JSON.
///
/// ```text
/// {
///   "version": 1,
///   "background": [r, g, b],
///   "triangles": [
///     {"verts": [[x, y], [x, y], [x, y]], "color": [r, g, b], "alpha": a}
///   ]
/// }
/// ```
pub fn scene_to_json(scene: &Scene) -> String {
    use std::fmt::Write as _;

    let mut s = String::from("{\n  \"version\": 1,\n  \"background\": ");
    push_array(&mut s, &scene.background);
    s.push_str(",\n  \"triangles\": [\n");

    for (i, t) in scene.tris.iter().enumerate() {
        s.push_str("    {\"verts\": [");
        for (j, v) in t.verts.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            push_array(&mut s, v);
        }
        s.push_str("], \"color\": ");
        push_array(&mut s, &t.color);
        let _ = write!(s, ", \"alpha\": {}}}", fmt_f32(t.alpha));
        if i + 1 < scene.tris.len() {
            s.push(',');
        }
        s.push('\n');
    }

    s.push_str("  ]\n}\n");
    s
}

fn push_array(s: &mut String, vals: &[f32]) {
    s.push('[');
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&fmt_f32(*v));
    }
    s.push(']');
}

/// Format a float as valid JSON.
///
/// `f32::to_string` emits `NaN` and `inf`, which are not JSON and would make
/// the output unparseable by every consumer. Parameters are sanitized during
/// fitting so these should not occur, but a serializer that can emit invalid
/// output is a trap waiting for the one case that slips through.
fn fmt_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "0.0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Triangle;

    fn sample() -> Scene {
        let mut s = Scene::new([0.1, 0.2, 0.3]);
        s.push(Triangle::new([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], [1.0, 0.5, 0.25], 0.75));
        s
    }

    #[test]
    fn emits_expected_fields() {
        let json = scene_to_json(&sample());
        assert!(json.contains("\"version\": 1"));
        assert!(json.contains("\"background\": [0.100000, 0.200000, 0.300000]"));
        assert!(json.contains("\"alpha\": 0.750000"));
    }

    #[test]
    fn empty_scene_is_still_valid_json() {
        let json = scene_to_json(&Scene::new([0.0; 3]));
        assert!(json.contains("\"triangles\": [\n  ]"));
    }

    #[test]
    fn non_finite_values_never_reach_the_output() {
        let mut s = sample();
        s.tris[0].alpha = f32::NAN;
        s.tris[0].verts[0][0] = f32::INFINITY;
        let json = scene_to_json(&s);
        assert!(!json.contains("NaN") && !json.contains("inf"), "{json}");
    }

    #[test]
    fn triangle_count_matches_separator_count() {
        let mut s = sample();
        s.push(s.tris[0].clone());
        s.push(s.tris[0].clone());
        let json = scene_to_json(&s);
        assert_eq!(json.matches("\"verts\"").count(), 3);
        // Two separators between three entries — a trailing comma is invalid JSON.
        assert_eq!(json.matches("},\n").count(), 2);
    }
}
