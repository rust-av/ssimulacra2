use std::f64::consts::PI;

use aligned::{Aligned, A16};
use nalgebra::base::{Matrix3, Matrix3x1};

const VERTICAL_MOD: usize = 4;
const VERTICAL_PREFETCH_ROWS: usize = 8;

pub struct Blur {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
}

impl Blur {
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
        self.kernel
            .fast_gaussian_vertical(&self.temp, &mut out, self.width, self.height);
        out
    }
}

/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016].
struct RecursiveGaussian {
    radius: usize,
    /// For k={1,3,5} in that order, each broadcasted 4x for LoadDup128. Used
    /// only for vertical passes.
    n2: Aligned<A16, [f32; 3 * 4]>,
    d1: Aligned<A16, [f32; 3 * 4]>,
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
            n2: Aligned(rg_n2),
            d1: Aligned(rg_d1),
            mul_prev: Aligned(mul_prev),
            mul_prev2: Aligned(mul_prev2),
            mul_in: Aligned(mul_in),
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn fast_gaussian_horizontal(&self, input: &[f32], output: &mut [f32], width: usize) {
        assert_eq!(input.len(), output.len());

        let mul_in_1 = self.mul_in[0];
        let mul_in_3 = self.mul_in[1];
        let mul_in_5 = self.mul_in[2];
        let mul_prev_1 = self.mul_prev[0];
        let mul_prev_3 = self.mul_prev[1];
        let mul_prev_5 = self.mul_prev[2];
        let mul_prev2_1 = self.mul_prev2[0];
        let mul_prev2_3 = self.mul_prev2[1];
        let mul_prev2_5 = self.mul_prev2[2];

        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let big_n = self.radius as isize;
        let mut small_n = (-big_n) + 1;

        while small_n < width as isize {
            let left = small_n - big_n - 1;
            let right = small_n + big_n - 1;
            let left_val = if left >= 0 {
                input[left as usize]
            } else {
                0f32
            };
            let right_val = if right < width as isize {
                input[right as usize]
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

            if small_n >= 0 {
                output[small_n as usize] = out_1 + out_3 + out_5;
            }

            small_n += 1;
        }
    }

    // Apply 1D vertical scan to multiple columns (one per vector lane).
    pub fn fast_gaussian_vertical(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert_eq!(input.len(), output.len());

        for x in 0..width {
            self.vertical_strip(input, x, output, width, height);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn vertical_strip(
        &self,
        input: &[f32],
        x: usize,
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        let d1_1 = self.d1[0];
        let d1_3 = self.d1[1];
        let d1_5 = self.d1[2];
        let n2_1 = self.n2[0];
        let n2_3 = self.n2[1];
        let n2_5 = self.n2[2];

        let big_n = self.radius as isize;
        let mut ctr = 0usize;
        let mut ring_buffer: Aligned<A16, _> = Aligned([0f32; 3 * VERTICAL_MOD]);

        let mut small_n = -(big_n as isize) + 1;

        // Warmup: top is out of bounds (zero padded), bottom is usually in-bounds.
        while small_n < 0 {
            // bottom is always non-negative since n is initialized in -N + 1.
            let bottom = (small_n + big_n - 1) as usize;
            vertical_block(
                d1_1,
                d1_3,
                d1_5,
                n2_1,
                n2_3,
                n2_5,
                &VertBlockInput::SingleInput(if bottom < height {
                    input[bottom * width + x]
                } else {
                    0f32
                }),
                &mut ctr,
                &mut ring_buffer,
                &mut VertBlockOutput::None,
            );
            small_n += 1;
        }

        // Start producing output; top is still out of bounds.
        while small_n < (big_n + 1).min(height as isize) {
            let bottom = (small_n + big_n - 1) as usize;
            vertical_block(
                d1_1,
                d1_3,
                d1_5,
                n2_1,
                n2_3,
                n2_5,
                &VertBlockInput::SingleInput(if bottom < height {
                    input[bottom * width + x]
                } else {
                    0f32
                }),
                &mut ctr,
                &mut ring_buffer,
                &mut VertBlockOutput::Store(&mut output[small_n as usize * width + x]),
            );
            small_n += 1;
        }

        // Interior outputs with prefetching and without bounds checks.
        while small_n < (height as isize - big_n + 1 - VERTICAL_PREFETCH_ROWS as isize) {
            let top = (small_n - big_n - 1) as usize;
            let bottom = (small_n + big_n - 1) as usize;
            vertical_block(
                d1_1,
                d1_3,
                d1_5,
                n2_1,
                n2_3,
                n2_5,
                &VertBlockInput::TwoInputs((input[top * width + x], input[bottom * width + x])),
                &mut ctr,
                &mut ring_buffer,
                &mut VertBlockOutput::Store(&mut output[small_n as usize * width + x]),
            );

            // TODO: Use https://doc.rust-lang.org/std/intrinsics/fn.prefetch_read_data.html when stabilized
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(target_arch = "x86")]
                use core::arch::x86::{_mm_prefetch, _MM_HINT_T0};
                #[cfg(target_arch = "x86_64")]
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                use std::ptr::addr_of;

                // SAFETY: We checked the target arch before calling this
                unsafe {
                    _mm_prefetch(
                        addr_of!(input[(top + VERTICAL_PREFETCH_ROWS) * width + x]).cast(),
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        addr_of!(input[(bottom + VERTICAL_PREFETCH_ROWS) * width + x]).cast(),
                        _MM_HINT_T0,
                    );
                }
            }

            small_n += 1;
        }

        // Bottom border without prefetching and with bounds checks.
        while small_n < height as isize {
            let top = (small_n - big_n - 1) as usize;
            let bottom = (small_n + big_n - 1) as usize;
            vertical_block(
                d1_1,
                d1_3,
                d1_5,
                n2_1,
                n2_3,
                n2_5,
                &VertBlockInput::TwoInputs((
                    input[top * width + x],
                    if bottom < height {
                        input[bottom * width + x]
                    } else {
                        0f32
                    },
                )),
                &mut ctr,
                &mut ring_buffer,
                &mut VertBlockOutput::Store(&mut output[small_n as usize * width + x]),
            );

            small_n += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vertical_block(
    d1_1: f32,
    d1_3: f32,
    d1_5: f32,
    n2_1: f32,
    n2_3: f32,
    n2_5: f32,
    input: &VertBlockInput,
    ctr: &mut usize,
    ring_buffer: &mut Aligned<A16, [f32; 3 * VERTICAL_MOD]>,
    output: &mut VertBlockOutput,
) {
    let (y_1, rest) = ring_buffer.split_at_mut(VERTICAL_MOD);
    let (y_3, y_5) = rest.split_at_mut(VERTICAL_MOD);

    *ctr += 1;
    let n_0 = (*ctr) % VERTICAL_MOD;
    let n_1 = (*ctr - 1) % VERTICAL_MOD;
    let n_2 = if *ctr == 1 {
        // During the first iteration, `ctr` can be 1, and for whatever reason `-1 % 4` in Rust
        // gives `-1` instead of the answer of `3` that it gives in C.
        // So we need to manually handle that case.
        (*ctr + 2) % VERTICAL_MOD
    } else {
        (*ctr - 2) % VERTICAL_MOD
    };

    let sum = input.get();

    let y_n1_1 = y_1[n_1];
    let y_n1_3 = y_3[n_1];
    let y_n1_5 = y_5[n_1];
    let y_n2_1 = y_1[n_2];
    let y_n2_3 = y_3[n_2];
    let y_n2_5 = y_5[n_2];

    let y1 = n2_1.mul_add(sum, neg_mul_sub(d1_1, y_n1_1, y_n2_1));
    let y3 = n2_3.mul_add(sum, neg_mul_sub(d1_3, y_n1_3, y_n2_3));
    let y5 = n2_5.mul_add(sum, neg_mul_sub(d1_5, y_n1_5, y_n2_5));
    y_1[n_0] = y1;
    y_3[n_0] = y3;
    y_5[n_0] = y5;
    output.write(y1 + y3 + y5);
}

enum VertBlockInput {
    SingleInput(f32),
    TwoInputs((f32, f32)),
}

impl VertBlockInput {
    pub fn get(&self) -> f32 {
        match *self {
            Self::SingleInput(input) => input,
            Self::TwoInputs((input1, input2)) => input1 + input2,
        }
    }
}

enum VertBlockOutput<'a> {
    None,
    Store(&'a mut f32),
}

impl<'a> VertBlockOutput<'a> {
    pub fn write(&mut self, data: f32) {
        match *self {
            Self::None => (),
            Self::Store(ref mut output) => {
                **output = data;
            }
        }
    }
}

#[inline(always)]
fn neg_mul_sub(mul: f32, x: f32, sub: f32) -> f32 {
    (-mul) * x - sub
}
