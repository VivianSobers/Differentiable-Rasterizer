//! WebAssembly bindings for the differentiable rasterizer.
//!
//! The browser cannot hand control to a function that blocks until a fit
//! converges — it has to yield between frames or the tab freezes. So this wraps
//! [`diffrast::Fitter`], the steppable form, and exposes a `step_many` the page
//! calls once per animation frame.
//!
//! Pixels cross the boundary as `Uint8ClampedArray` in RGBA order, which is
//! exactly what `ImageData` wants, so the page can blit results to a canvas
//! with no conversion on the JavaScript side.

use diffrast::canvas::Canvas;
use diffrast::raster::{render, RenderParams};
use diffrast::{scene_to_json, FitConfig, Fitter};
use wasm_bindgen::prelude::*;

/// Install a panic hook that reports Rust panics to the browser console.
///
/// Without this a panic in WebAssembly surfaces as `unreachable executed` with
/// no location, which is close to undebuggable.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// A fit running in the browser.
#[wasm_bindgen]
pub struct WasmFitter {
    fitter: Fitter,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmFitter {
    /// Create a fitter for an RGBA image.
    ///
    /// `rgba` must be `width * height * 4` bytes in sRGB, straight from a
    /// canvas `getImageData` call. Returns a JS error rather than panicking on
    /// bad input, so the page can show a message instead of dying.
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        rgba: &[u8],
        triangles: usize,
        iters: usize,
        seed: u32,
    ) -> Result<WasmFitter, JsValue> {
        Self::build(width, height, rgba, triangles, iters, seed).map_err(js_err)
    }

    /// Run up to `n` iterations, returning how many actually ran.
    ///
    /// Batched because a single iteration is far shorter than a frame, and
    /// crossing the wasm boundary once per iteration would cost more than the
    /// work itself.
    pub fn step_many(&mut self, n: usize) -> usize {
        let mut done = 0;
        for _ in 0..n {
            if self.fitter.step().is_none() {
                break;
            }
            done += 1;
        }
        done
    }

    /// The current scene rendered as RGBA, ready for `ImageData`.
    pub fn render_rgba(&self) -> Vec<u8> {
        rgba_from_canvas(&self.fitter.render_current())
    }

    /// The current scene rendered at an arbitrary size, for sharp export.
    pub fn render_rgba_at(&self, width: usize, height: usize) -> Result<Vec<u8>, JsValue> {
        self.try_render_rgba_at(width, height).map_err(js_err)
    }

    #[wasm_bindgen(getter)]
    pub fn iter(&self) -> usize {
        self.fitter.iter()
    }

    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        self.fitter.is_done()
    }

    #[wasm_bindgen(getter)]
    pub fn loss(&self) -> f32 {
        self.fitter.losses().last().copied().unwrap_or(f32::NAN)
    }

    #[wasm_bindgen(getter)]
    pub fn best_loss(&self) -> f32 {
        self.fitter.best_loss()
    }

    #[wasm_bindgen(getter)]
    pub fn sigma(&self) -> f32 {
        self.fitter.sigma()
    }

    #[wasm_bindgen(getter)]
    pub fn triangles(&self) -> usize {
        self.fitter.scene().len()
    }

    /// Every loss so far, for plotting.
    pub fn losses(&self) -> Vec<f32> {
        self.fitter.losses().to_vec()
    }

    /// The current scene as JSON, in the same format the CLI writes.
    pub fn scene_json(&self) -> String {
        scene_to_json(self.fitter.scene())
    }
}

/// Fallible operations, with plain-`String` errors.
///
/// Split out from the `#[wasm_bindgen]` surface because `JsValue` can only be
/// constructed inside a wasm runtime — building one on a native target aborts.
/// Keeping the logic here means every validation path is covered by ordinary
/// `cargo test`, and the bindings above stay a thin error-conversion shim.
impl WasmFitter {
    pub(crate) fn build(
        width: usize,
        height: usize,
        rgba: &[u8],
        triangles: usize,
        iters: usize,
        seed: u32,
    ) -> Result<WasmFitter, String> {
        if width == 0 || height == 0 {
            return Err("image dimensions must be non-zero".to_string());
        }
        let expected = width * height * 4;
        if rgba.len() != expected {
            return Err(format!("expected {expected} bytes of RGBA, got {}", rgba.len()));
        }

        let target = canvas_from_rgba(width, height, rgba);
        let cfg = FitConfig {
            triangles,
            iters,
            seed: seed as u64,
            // The viewer is a demo: a user watching triangles converge wants to
            // see the whole run, not have it stop quietly two thirds through.
            patience: None,
            ..Default::default()
        };

        let fitter = Fitter::new(target, cfg).map_err(|e| e.to_string())?;
        Ok(WasmFitter { fitter, width, height })
    }

    pub(crate) fn try_render_rgba_at(
        &self,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, String> {
        if width == 0 || height == 0 {
            return Err("dimensions must be non-zero".to_string());
        }
        // Scale sigma with resolution so an upscaled render keeps the same
        // apparent edge softness rather than turning blurry.
        let sigma = 0.0015 * self.width as f32 / width as f32;
        let canvas = render(self.fitter.scene(), RenderParams::new(width, height, sigma));
        Ok(rgba_from_canvas(&canvas))
    }
}

fn js_err(msg: String) -> JsValue {
    JsValue::from_str(&msg)
}

/// Decode sRGB bytes into the linear-light canvas the renderer works in.
fn canvas_from_rgba(width: usize, height: usize, rgba: &[u8]) -> Canvas {
    let mut canvas = Canvas::new(width, height);
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        for ch in 0..3 {
            canvas.data[i * 3 + ch] = srgb_to_linear(px[ch]);
        }
    }
    canvas
}

/// Encode the linear canvas back to sRGB bytes, fully opaque.
fn rgba_from_canvas(canvas: &Canvas) -> Vec<u8> {
    let mut out = Vec::with_capacity(canvas.width * canvas.height * 4);
    for px in canvas.data.chunks_exact(3) {
        out.push(linear_to_srgb(px[0]));
        out.push(linear_to_srgb(px[1]));
        out.push(linear_to_srgb(px[2]));
        out.push(255);
    }
    out
}

#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_rgba(w: usize, h: usize, level: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&[level, level, level, 255]);
        }
        v
    }

    #[test]
    fn srgb_conversion_round_trips() {
        for v in [0u8, 1, 64, 128, 255] {
            assert_eq!(linear_to_srgb(srgb_to_linear(v)), v);
        }
    }

    #[test]
    fn rejects_mismatched_buffer_length() {
        let err = WasmFitter::build(4, 4, &gray_rgba(2, 2, 128), 4, 10, 0);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert!(WasmFitter::build(0, 4, &[], 4, 10, 0).is_err());
    }

    #[test]
    fn rejects_invalid_triangle_count() {
        assert!(WasmFitter::build(4, 4, &gray_rgba(4, 4, 128), 0, 10, 0).is_err());
    }

    #[test]
    fn steps_and_reports_progress() {
        let mut f = WasmFitter::build(8, 8, &gray_rgba(8, 8, 100), 4, 10, 0).unwrap();
        assert_eq!(f.step_many(4), 4);
        assert_eq!(f.iter(), 4);
        assert!(f.loss().is_finite());
        assert_eq!(f.triangles(), 4);
        assert_eq!(f.losses().len(), 4);
    }

    #[test]
    fn stops_at_the_iteration_budget() {
        let mut f = WasmFitter::build(8, 8, &gray_rgba(8, 8, 100), 4, 5, 0).unwrap();
        // Asking for more than remain must return the real count, not the ask.
        assert_eq!(f.step_many(50), 5);
        assert!(f.done());
        assert_eq!(f.step_many(10), 0);
    }

    #[test]
    fn renders_at_the_requested_size() {
        let f = WasmFitter::build(8, 8, &gray_rgba(8, 8, 100), 4, 5, 0).unwrap();
        assert_eq!(f.render_rgba().len(), 8 * 8 * 4);
        assert_eq!(f.try_render_rgba_at(32, 16).unwrap().len(), 32 * 16 * 4);
        assert!(f.try_render_rgba_at(0, 16).is_err());
    }

    #[test]
    fn exports_scene_json() {
        let f = WasmFitter::build(8, 8, &gray_rgba(8, 8, 100), 3, 5, 0).unwrap();
        let json = f.scene_json();
        assert!(json.contains("\"version\": 1"));
        assert_eq!(json.matches("\"verts\"").count(), 3);
    }

    #[test]
    fn alpha_channel_is_always_opaque() {
        let f = WasmFitter::build(4, 4, &gray_rgba(4, 4, 200), 2, 3, 0).unwrap();
        assert!(f.render_rgba().chunks_exact(4).all(|px| px[3] == 255));
    }
}
