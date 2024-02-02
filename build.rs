use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;

use nalgebra::{Matrix3, Matrix3x1};

fn main() {
    let out_dir = &env::var("OUT_DIR").expect("can read OUT_DIR");

    init_recursive_gaussian(out_dir).expect("can init recursive gaussian");
}

fn write_const_f32<W: Write>(w: &mut W, name: &str, val: f32) -> io::Result<()> {
    writeln!(w, "const {name}: f32 = {val}_f32;")
}

fn write_const_usize<W: Write>(w: &mut W, name: &str, val: usize) -> io::Result<()> {
    writeln!(w, "const {name}: usize = {val}_usize;")
}

fn init_recursive_gaussian(out_path: &str) -> io::Result<()> {
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
    let d_13 = p_1.mul_add(r_3, -r_1 * p_3);
    let d_35 = p_3.mul_add(r_5, -r_3 * p_5);
    let d_51 = p_5.mul_add(r_1, -r_5 * p_1);

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
        radius.mul_add(radius, -SIGMA * SIGMA),
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
        mul_prev[4 * i + 3] = d_2.mul_add(d_2, 3.0f64.mul_add(-d_2, 1.0f64)) as f32;
        mul_prev2[4 * i] = -1.0f32;
        mul_prev2[4 * i + 1] = d1[i] as f32;
        mul_prev2[4 * i + 2] = (-d_2 + 1.0f64) as f32;
        mul_prev2[4 * i + 3] = d_2.mul_add(d1[i], -2.0f64 * d1[i]) as f32;
        mul_in[4 * i] = n2[i] as f32;
        mul_in[4 * i + 1] = (-d1[i] * n2[i]) as f32;
        mul_in[4 * i + 2] = d_2.mul_add(n2[i], -n2[i]) as f32;
        mul_in[4 * i + 3] = (-d_2 * d1[i]).mul_add(n2[i], 2.0f64 * d1[i] * n2[i]) as f32;
    }

    let file_path = Path::new(out_path).join("recursive_gaussian.rs");
    let mut out_file = File::create(file_path)?;

    write_const_usize(&mut out_file, "RADIUS", radius as usize)?;

    write_const_f32(&mut out_file, "VERT_MUL_IN_1", n2[0] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_IN_3", n2[1] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_IN_5", n2[2] as f32)?;

    write_const_f32(&mut out_file, "VERT_MUL_PREV_1", d1[0] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_PREV_3", d1[1] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_PREV_5", d1[2] as f32)?;

    write_const_f32(&mut out_file, "MUL_IN_1", mul_in[0])?;
    write_const_f32(&mut out_file, "MUL_IN_3", mul_in[4])?;
    write_const_f32(&mut out_file, "MUL_IN_5", mul_in[8])?;

    write_const_f32(&mut out_file, "MUL_PREV_1", mul_prev[0])?;
    write_const_f32(&mut out_file, "MUL_PREV_3", mul_prev[4])?;
    write_const_f32(&mut out_file, "MUL_PREV_5", mul_prev[8])?;

    write_const_f32(&mut out_file, "MUL_PREV2_1", mul_prev2[0])?;
    write_const_f32(&mut out_file, "MUL_PREV2_3", mul_prev2[4])?;
    write_const_f32(&mut out_file, "MUL_PREV2_5", mul_prev2[8])?;

    Ok(())
}
