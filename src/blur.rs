use std::f64::consts::PI;

use nalgebra::base::{Matrix3, Matrix3x1};

pub struct Blur {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
}

impl Blur {
    pub fn new(width: usize, height: usize) -> Self {
        Blur {
            kernel: RecursiveGaussian::new(1.5),
            temp: vec![0.0f32; width * height],
            width,
            height,
        }
    }

    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.temp.truncate(width * height);
        self.width = width;
        self.height = height;
    }
}

/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016].
struct RecursiveGaussian {
    radius: usize,
    /// For k={1,3,5} in that order, each broadcasted 4x for LoadDup128. Used
    /// only for vertical passes.
    n2: [f32; 3 * 4],
    d1: [f32; 3 * 4],
    /// We unroll horizontal passes 4x - one output per lane. These are each
    /// lane's multiplier for the previous output (relative to the first of
    /// the four outputs). Indexing: 4 * 0..2 (for {1,3,5}) + 0..3 for the
    /// lane index.
    mul_prev: [f32; 3 * 4],
    /// Ditto for the second to last output.
    mul_prev2: [f32; 3 * 4],
    /// We multiply a vector of inputs 0..3 by a vector shifted from this array.
    /// in=0 uses all 4 (nonzero) terms; for in=3, the lower three lanes are 0.
    mul_in: [f32; 3 * 4],
}

impl RecursiveGaussian {
    pub fn new(sigma: f64) -> Self {
        // (57), "N"
        let radius = 3.2795f64.mul_add(sigma, 0.2546);

        // Table I, first row
        let pi_div_2r = PI / (2.0f64 * radius);
        let omega = [pi_div_2r, 3.0f64 * pi_div_2r, 5.0f64 * pi_div_2r];

        // (37), k={1,3,5}
        let p_1 = 1.0f64 / (0.5 * omega[0]).tan();
        let p_3 = -1.0f64 / (0.5 * omega[1]).tan();
        let p_5 = 1.0f64 / (0.5 * omega[2]).tan();

        // (44), k={1,3,5}
        let r_1 = p_1 * p_1 / omega[0].sin();
        let r_3 = -p_3 * p_3 / omega[1].sin();
        let r_5 = p_5 * p_5 / omega[2].sin();

        // (50), k={1,3,5}
        let neg_half_sigma2 = -0.5f64 * sigma * sigma;
        let recip_radius = 1.0f64 / radius;
        let mut rho = [0.0f64; 3];
        for i in 0..3 {
            rho[i] = (neg_half_sigma2 * omega[i] * omega[i]).exp() * recip_radius;
        }

        // second part of (52), k1,k2 = 1,3; 3,5; 5,1
        let d_13 = p_1 * r_3 - r_1 * p_3;
        let d_35 = p_3 * r_5 - r_3 * p_5;
        let d_51 = p_5 * r_1 - r_5 * p_1;

        // (52), k=5
        let recip_d13 = 1.0f64 / d_13;
        let zeta_15 = d_35 * recip_d13;
        let zeta_35 = d_51 * recip_d13;

        // (56)
        let a = Matrix3::from_row_slice(&[p_1, p_3, p_5, r_1, r_3, r_5, zeta_15, zeta_35, 1.0f64]);
        assert!(a.try_inverse().is_some());
        // (55)
        let gamma = Matrix3x1::from_column_slice(&[
            1.0f64,
            radius * radius - sigma * sigma,
            zeta_15.mul_add(rho[0], zeta_35 * rho[1]) + rho[2],
        ]);
        // (53)
        let beta = a * gamma;

        // Sanity check: correctly solved for beta (IIR filter weights are normalized)
        // (39)
        let sum = beta[2].mul_add(p_5, beta[0].mul_add(p_1, beta[1] * p_3));
        assert!((sum - 1.0).abs() < 1E-12f64);

        let mut n2 = [0f64; 3];
        let mut d1 = [0f64; 3];
        let mut rg_n2 = [0f32; 3 * 4];
        let mut rg_d1 = [0f32; 3 * 4];
        let mut mul_prev = [0f32; 3 * 4];
        let mut mul_prev2 = [0f32; 3 * 4];
        let mut mul_in = [0f32; 3 * 4];
        for i in 0..3 {
            // (33)
            n2[i] = -beta[i] * (omega[i] * (radius + 1.0)).cos();
            d1[i] = -2.0f64 * omega[i].cos();

            for lane in 0..4 {
                rg_n2[4 * i + lane] = n2[i] as f32;
                rg_d1[4 * i + lane] = d1[i] as f32;
            }

            let d_2 = d1[i] * d1[i];

            // Obtained by expanding (35) for four consecutive outputs via
            // sympy: n, d, p, pp = symbols('n d p pp')
            // i0, i1, i2, i3 = symbols('i0 i1 i2 i3')
            // o0, o1, o2, o3 = symbols('o0 o1 o2 o3')
            // o0 = n*i0 - d*p - pp
            // o1 = n*i1 - d*o0 - p
            // o2 = n*i2 - d*o1 - o0
            // o3 = n*i3 - d*o2 - o1
            // Then expand(o3) and gather terms for p(prev), pp(prev2) etc.
            mul_prev[4 * i] = -d1[i] as f32;
            mul_prev[4 * i + 1] = (d_2 - 1.0f64) as f32;
            mul_prev[4 * i + 2] = (-d_2).mul_add(d1[i], 2.0f64 * d1[i]) as f32;
            mul_prev[4 * i + 3] = (d_2 * d_2 - 3.0f64 * d_2 + 1.0f64) as f32;
            mul_prev2[4 * i] = -1.0f32;
            mul_prev2[4 * i + 1] = d1[i] as f32;
            mul_prev2[4 * i + 2] = (-d_2 + 1.0f64) as f32;
            mul_prev2[4 * i + 3] = (d_2 * d1[i] - 2.0f64 * d1[i]) as f32;
            mul_in[4 * i] = n2[i] as f32;
            mul_in[4 * i + 1] = (-d1[i] * n2[i]) as f32;
            mul_in[4 * i + 2] = (d_2 * n2[i] - n2[i]) as f32;
            mul_in[4 * i + 3] = (-d_2 * d1[i]).mul_add(n2[i], 2.0f64 * d1[i] * n2[i]) as f32;
        }

        Self {
            radius: radius as usize,
            n2: rg_n2,
            d1: rg_d1,
            mul_prev,
            mul_prev2,
            mul_in,
        }
    }
}
