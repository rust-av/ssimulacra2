use std::f64::consts::PI;

use aligned::{Aligned, A16};
use nalgebra::base::{Matrix3, Matrix3x1};

pub struct Blur {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
}

impl Blur {
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Blur {
            kernel: RecursiveGaussian::new(),
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

    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.kernel
            .fast_gaussian_horizontal(plane, &mut self.temp, self.width);
        self.kernel.fast_gaussian_vertical_chunked::<128, 32>(
            &self.temp,
            &mut out,
            self.width,
            self.height,
        );
        out
    }
}

/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016].
struct RecursiveGaussian {
    radius: usize,
    /// For k={1,3,5} in that order.
    vert_mul_in: [f32; 3],
    vert_mul_prev: [f32; 3],
    /// We unroll horizontal passes 4x - one output per lane. These are each
    /// lane's multiplier for the previous output (relative to the first of
    /// the four outputs). Indexing: 4 * 0..2 (for {1,3,5}) + 0..3 for the
    /// lane index.
    mul_prev: Aligned<A16, [f32; 3 * 4]>,
    /// Ditto for the second to last output.
    mul_prev2: Aligned<A16, [f32; 3 * 4]>,
    /// We multiply a vector of inputs 0..3 by a vector shifted from this array.
    /// in=0 uses all 4 (nonzero) terms; for in=3, the lower three lanes are 0.
    mul_in: Aligned<A16, [f32; 3 * 4]>,
}

impl RecursiveGaussian {
    pub fn new() -> Self {
        const SIGMA: f64 = 1.5f64;

        // (57), "N"
        let radius = 3.2795f64.mul_add(SIGMA, 0.2546).round();

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
        let neg_half_sigma2 = -0.5f64 * SIGMA * SIGMA;
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
        let a = Matrix3::from_row_slice(&[p_1, p_3, p_5, r_1, r_3, r_5, zeta_15, zeta_35, 1.0f64])
            .try_inverse()
            .expect("Has inverse");
        // (55)
        let gamma = Matrix3x1::from_column_slice(&[
            1.0f64,
            radius * radius - SIGMA * SIGMA,
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
        let mut mul_prev = [0f32; 3 * 4];
        let mut mul_prev2 = [0f32; 3 * 4];
        let mut mul_in = [0f32; 3 * 4];
        for i in 0..3 {
            // (33)
            n2[i] = -beta[i] * (omega[i] * (radius + 1.0)).cos();
            d1[i] = -2.0f64 * omega[i].cos();

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
            vert_mul_in: n2.map(|f| f as f32),
            vert_mul_prev: d1.map(|f| f as f32),
            mul_prev: Aligned(mul_prev),
            mul_prev2: Aligned(mul_prev2),
            mul_in: Aligned(mul_in),
        }
    }

    #[cfg(feature = "rayon")]
    pub fn fast_gaussian_horizontal(&self, input: &[f32], output: &mut [f32], width: usize) {
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::prelude::ParallelSliceMut;
        use rayon::slice::ParallelSlice;

        assert_eq!(input.len(), output.len());

        input
            .par_chunks_exact(width)
            .zip(output.par_chunks_exact_mut(width))
            .for_each(|(input, output)| self.horizontal_row(input, output, width));
    }

    fn horizontal_row(&self, input: &[f32], output: &mut [f32], width: usize) {
        let big_n = self.radius as isize;
        let mul_in_1 = self.mul_in[0];
        let mul_in_3 = self.mul_in[4];
        let mul_in_5 = self.mul_in[8];
        let mul_prev_1 = self.mul_prev[0];
        let mul_prev_3 = self.mul_prev[4];
        let mul_prev_5 = self.mul_prev[8];
        let mul_prev2_1 = self.mul_prev2[0];
        let mul_prev2_3 = self.mul_prev2[4];
        let mul_prev2_5 = self.mul_prev2[8];
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let mut n = (-big_n) + 1;
        while n < width as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;
            let left_val = if left >= 0 {
                // SAFETY: `left` can never be bigger than `width`
                unsafe { *input.get_unchecked(left as usize) }
            } else {
                0f32
            };
            let right_val = if right < width as isize {
                // SAFETY: this branch ensures that `right` is not bigger than `width`
                unsafe { *input.get_unchecked(right as usize) }
            } else {
                0f32
            };
            let sum = left_val + right_val;

            let mut out_1 = sum * mul_in_1;
            let mut out_3 = sum * mul_in_3;
            let mut out_5 = sum * mul_in_5;

            out_1 = mul_prev2_1.mul_add(prev2_1, out_1);
            out_3 = mul_prev2_3.mul_add(prev2_3, out_3);
            out_5 = mul_prev2_5.mul_add(prev2_5, out_5);
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;

            out_1 = mul_prev_1.mul_add(prev_1, out_1);
            out_3 = mul_prev_3.mul_add(prev_3, out_3);
            out_5 = mul_prev_5.mul_add(prev_5, out_5);
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 {
                // SAFETY: We know that this chunk of output is of size `width`,
                // which `n` cannot be larger than.
                unsafe {
                    *output.get_unchecked_mut(n as usize) = out_1 + out_3 + out_5;
                }
            }

            n += 1;
        }
    }

    pub fn fast_gaussian_vertical_chunked<const J: usize, const K: usize>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert!(J > K);
        assert!(K > 0);

        assert_eq!(input.len(), output.len());

        let mut x = 0;
        while x + J <= width {
            self.fast_gaussian_vertical::<J>(&input[x..], &mut output[x..], width, height);
            x += J;
        }

        while x + K <= width {
            self.fast_gaussian_vertical::<K>(&input[x..], &mut output[x..], width, height);
            x += K;
        }

        while x < width {
            self.fast_gaussian_vertical::<1>(&input[x..], &mut output[x..], width, height);
            x += 1;
        }
    }

    // Apply 1D vertical scan on COLUMNS elements at a time
    pub fn fast_gaussian_vertical<const COLUMNS: usize>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert_eq!(input.len(), output.len());

        let big_n = self.radius as isize;

        let zeroes = vec![0f32; COLUMNS];
        let mut prev = vec![0f32; 3 * COLUMNS];
        let mut prev2 = vec![0f32; 3 * COLUMNS];
        let mut out = vec![0f32; 3 * COLUMNS];

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let top_row = if top >= 0 {
                &input[top as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            let bottom_row = if bottom < height as isize {
                &input[bottom as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            for i in 0..COLUMNS {
                let sum = top_row[i] + bottom_row[i];

                let i1 = i;
                let i3 = i1 + COLUMNS;
                let i5 = i3 + COLUMNS;

                let mp1 = self.vert_mul_prev[0];
                let mp3 = self.vert_mul_prev[1];
                let mp5 = self.vert_mul_prev[2];

                let out1 = prev[i1].mul_add(mp1, prev2[i1]);
                let out3 = prev[i3].mul_add(mp3, prev2[i3]);
                let out5 = prev[i5].mul_add(mp5, prev2[i5]);

                let mi1 = self.vert_mul_in[0];
                let mi3 = self.vert_mul_in[1];
                let mi5 = self.vert_mul_in[2];

                let out1 = sum.mul_add(mi1, -out1);
                let out3 = sum.mul_add(mi3, -out3);
                let out5 = sum.mul_add(mi5, -out5);

                out[i1] = out1;
                out[i3] = out3;
                out[i5] = out5;

                if n >= 0 {
                    output[n as usize * width + i] = out1 + out3 + out5;
                }
            }

            prev2.copy_from_slice(&prev);
            prev.copy_from_slice(&out);

            n += 1;
        }
    }
}
