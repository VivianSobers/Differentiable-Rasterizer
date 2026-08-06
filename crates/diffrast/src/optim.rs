//! Adam, with a per-parameter learning rate.
//!
//! The per-parameter rate is not a generality flourish — it is required here.
//! Vertex positions, colors, and alpha live on different scales and respond at
//! very different speeds; a single global rate that moves geometry usefully
//! will send colors oscillating, and one tuned for colors leaves triangles
//! effectively frozen in place.

/// Adam optimizer state over a flat parameter vector.
#[derive(Clone, Debug)]
pub struct Adam {
    /// First moment (running mean of gradients).
    m: Vec<f32>,
    /// Second moment (running mean of squared gradients).
    v: Vec<f32>,
    /// Step count, used for bias correction.
    t: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Adam {
    pub fn new(n_params: usize) -> Self {
        Self { m: vec![0.0; n_params], v: vec![0.0; n_params], t: 0, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }

    pub fn steps_taken(&self) -> u32 {
        self.t
    }

    /// Apply one update in place. `lr` supplies a learning rate per parameter.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: &[f32]) {
        assert_eq!(params.len(), grads.len(), "gradient length mismatch");
        assert_eq!(params.len(), lr.len(), "learning-rate length mismatch");

        self.t += 1;
        // Bias correction: both moments start at zero, so early steps would
        // otherwise be biased toward zero and the fit would crawl for the first
        // few dozen iterations.
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr[i] * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimize `(x - 3)^2`, whose gradient is `2(x - 3)`.
    #[test]
    fn descends_a_quadratic() {
        let mut adam = Adam::new(1);
        let mut p = vec![0.0f32];
        let lr = vec![0.1f32];
        for _ in 0..500 {
            let g = vec![2.0 * (p[0] - 3.0)];
            adam.step(&mut p, &g, &lr);
        }
        assert!((p[0] - 3.0).abs() < 1e-2, "converged to {}", p[0]);
        assert_eq!(adam.steps_taken(), 500);
    }

    #[test]
    fn zero_gradient_leaves_params_untouched() {
        let mut adam = Adam::new(3);
        let mut p = vec![1.0, -2.0, 0.5];
        let before = p.clone();
        adam.step(&mut p, &[0.0; 3], &[0.1; 3]);
        assert_eq!(p, before);
    }

    #[test]
    fn per_parameter_rates_are_respected() {
        let mut adam = Adam::new(2);
        let mut p = vec![0.0f32, 0.0];
        // Identical gradients, different rates: the first should move further.
        adam.step(&mut p, &[1.0, 1.0], &[0.5, 0.01]);
        assert!(p[0].abs() > p[1].abs() * 10.0, "got {p:?}");
    }
}
