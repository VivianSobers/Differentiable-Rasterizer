//! A differentiable 2D triangle rasterizer.
//!
//! Rendering is deliberately *soft*: a pixel's coverage by a triangle is a
//! sigmoid of the signed distance to the triangle boundary rather than a hard
//! in/out test. That single change is what makes the whole pipeline
//! differentiable — a hard test has zero gradient everywhere and an undefined
//! one exactly at the edge, so an optimizer never learns which way to move a
//! vertex.

pub mod canvas;
pub mod grad;
pub mod raster;
pub mod scene;

pub use canvas::Canvas;
pub use grad::{backward, finite_difference, render_with_tape, Tape};
pub use raster::{render, RenderParams};
pub use scene::{Scene, Triangle};
