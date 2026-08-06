use std::path::Path;

/// A flat RGB f32 image. Kept in linear space during rendering; gamma is only
/// applied on the way out to a PNG.
#[derive(Clone, Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` values, row-major.
    pub data: Vec<f32>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height * 3] }
    }

    pub fn filled(width: usize, height: usize, color: [f32; 3]) -> Self {
        let mut c = Self::new(width, height);
        for px in c.data.chunks_exact_mut(3) {
            px.copy_from_slice(&color);
        }
        c
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        (y * self.width + x) * 3
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 3] {
        let i = self.idx(x, y);
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, c: [f32; 3]) {
        let i = self.idx(x, y);
        self.data[i..i + 3].copy_from_slice(&c);
    }

    /// Mean squared error against another canvas of the same size — the loss
    /// the fitting loop minimizes.
    pub fn mse(&self, other: &Canvas) -> f32 {
        assert_eq!(self.data.len(), other.data.len(), "canvas size mismatch");
        // Accumulated in f64: an f32 sum over a megapixel of small residuals
        // loses enough precision to swamp a finite-difference gradient check.
        let sum: f64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| {
                let d = (a - b) as f64;
                d * d
            })
            .sum();
        (sum / self.data.len() as f64) as f32
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), image::ImageError> {
        let mut buf = image::RgbImage::new(self.width as u32, self.height as u32);
        for (i, px) in buf.pixels_mut().enumerate() {
            let c = &self.data[i * 3..i * 3 + 3];
            *px = image::Rgb([encode_srgb(c[0]), encode_srgb(c[1]), encode_srgb(c[2])]);
        }
        buf.save(path)
    }

    /// Load a PNG/JPEG into linear space and resize to the requested size.
    pub fn load_image(
        path: impl AsRef<Path>,
        width: usize,
        height: usize,
    ) -> Result<Self, image::ImageError> {
        let img = image::open(path)?
            .resize_exact(width as u32, height as u32, image::imageops::FilterType::Lanczos3)
            .to_rgb8();
        let mut c = Canvas::new(width, height);
        for (i, px) in img.pixels().enumerate() {
            for ch in 0..3 {
                c.data[i * 3 + ch] = decode_srgb(px.0[ch]);
            }
        }
        Ok(c)
    }
}

#[inline]
fn encode_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

#[inline]
fn decode_srgb(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
