#![deny(clippy::all)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::inconsistent_struct_constructor)]
#![allow(clippy::inline_always)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::use_self)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::create_dir)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::exit)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::mem_forget)]
#![warn(clippy::mod_module_files)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::pattern_type_mismatch)]
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::same_name_method)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::use_debug)]
#![warn(clippy::verbose_file_reads)]

mod blur;

use anyhow::{bail, Result};
use blur::Blur;
pub use yuvxyb::{CastFromPrimitive, Frame, LinearRgb, Pixel, Plane, Rgb, Xyb, Yuv};
pub use yuvxyb::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic, YuvConfig};

const NUM_SCALES: usize = 6;

/// Computes the SSIMULACRA2 score for a given input frame and the distorted
/// version of that frame.
///
/// # Errors
/// - If the source and distorted image width and height do not match
/// - If the source or distorted image cannot be converted to XYB successfully
/// - If the image is smaller than 8x8 pixels
pub fn compute_frame_ssimulacra2<T: TryInto<LinearRgb>, U: TryInto<LinearRgb>>(
    source: T,
    distorted: U,
) -> Result<f64> {
    let mut img1: LinearRgb = source
        .try_into()
        .map_err(|_e| anyhow::anyhow!("Failed to convert to Linear Rgb"))?;
    let mut img2: LinearRgb = distorted
        .try_into()
        .map_err(|_e| anyhow::anyhow!("Failed to convert to Linear Rgb"))?;

    if img1.width() != img2.width() || img1.height() != img2.height() {
        bail!("Source and distorted image width and height must be equal");
    }

    if img1.width() < 8 || img1.height() < 8 {
        bail!("Images must be at least 8x8 pixels");
    }

    let mut width = img1.width();
    let mut height = img1.height();

    let mut mul = [
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
    ];
    let mut blur = Blur::new(width, height);
    let mut msssim = Msssim::default();

    for scale in 0..NUM_SCALES {
        if width < 8 || height < 8 {
            break;
        }

        if scale > 0 {
            img1 = downscale_by_2(&img1);
            img2 = downscale_by_2(&img2);
            width = img1.width();
            height = img2.height();
        }
        for c in &mut mul {
            c.truncate(width * height);
        }
        blur.shrink_to(width, height);

        let mut img1 = Xyb::from(&img1);
        let mut img2 = Xyb::from(&img2);

        make_positive_xyb(&mut img1);
        make_positive_xyb(&mut img2);

        // SSIMULACRA2 works with the data in a planar format,
        // so we need to convert to that.
        let img1 = xyb_to_planar(&img1);
        let img2 = xyb_to_planar(&img2);

        image_multiply(&img1, &img1, &mut mul);
        let sigma1_sq = blur.blur(&mul);

        image_multiply(&img2, &img2, &mut mul);
        let sigma2_sq = blur.blur(&mul);

        image_multiply(&img1, &img2, &mut mul);
        let sigma12 = blur.blur(&mul);

        let mu1 = blur.blur(&img1);
        let mu2 = blur.blur(&img2);

        let avg_ssim = ssim_map(width, height, &mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12);
        let avg_edgediff = edge_diff_map(width, height, &img1, &mu1, &img2, &mu2);
        msssim.scales.push(MsssimScale {
            avg_ssim,
            avg_edgediff,
        });
    }

    Ok(msssim.score())
}

fn make_positive_xyb(xyb: &mut Xyb) {
    for pix in xyb.data_mut().iter_mut() {
        pix[2] += 1.1f32 - pix[1];
        pix[0] += 0.5f32;
        pix[1] += 0.05f32;
    }
}

fn xyb_to_planar(xyb: &Xyb) -> [Vec<f32>; 3] {
    let mut out1 = vec![0.0f32; xyb.width() * xyb.height()];
    let mut out2 = vec![0.0f32; xyb.width() * xyb.height()];
    let mut out3 = vec![0.0f32; xyb.width() * xyb.height()];
    for (((i, o1), o2), o3) in xyb
        .data()
        .iter()
        .copied()
        .zip(out1.iter_mut())
        .zip(out2.iter_mut())
        .zip(out3.iter_mut())
    {
        *o1 = i[0];
        *o2 = i[1];
        *o3 = i[2];
    }

    [out1, out2, out3]
}

fn image_multiply(img1: &[Vec<f32>; 3], img2: &[Vec<f32>; 3], out: &mut [Vec<f32>; 3]) {
    for ((plane1, plane2), out_plane) in img1.iter().zip(img2.iter()).zip(out.iter_mut()) {
        for ((&p1, &p2), o) in plane1.iter().zip(plane2.iter()).zip(out_plane.iter_mut()) {
            *o = p1 * p2;
        }
    }
}

fn downscale_by_2(in_data: &LinearRgb) -> LinearRgb {
    const SCALE: usize = 2;
    let in_w = in_data.width();
    let in_h = in_data.height();
    let out_w = (in_w + SCALE - 1) / SCALE;
    let out_h = (in_h + SCALE - 1) / SCALE;
    let mut out_data = vec![[0.0f32; 3]; out_w * out_h];
    let normalize = 1f32 / (SCALE * SCALE) as f32;

    let in_data = &in_data.data();
    for oy in 0..out_h {
        for ox in 0..out_w {
            for c in 0..3 {
                let mut sum = 0f32;
                for iy in 0..SCALE {
                    for ix in 0..SCALE {
                        let x = (ox * SCALE + ix).min(in_w - 1);
                        let y = (oy * SCALE + iy).min(in_h - 1);
                        let in_pix = in_data[y * in_w + x];

                        sum += in_pix[c];
                    }
                }
                let out_pix = &mut out_data[oy * out_w + ox];
                out_pix[c] = sum * normalize;
            }
        }
    }

    LinearRgb::new(out_data, out_w, out_h).expect("Resolution and data size match")
}

fn ssim_map(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    const C2: f32 = 0.0009f32;

    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    for c in 0..3 {
        let mut sum1 = [0.0f64; 2];
        for (row_m1, (row_m2, (row_s11, (row_s22, row_s12)))) in m1[c].chunks_exact(width).zip(
            m2[c].chunks_exact(width).zip(
                s11[c]
                    .chunks_exact(width)
                    .zip(s22[c].chunks_exact(width).zip(s12[c].chunks_exact(width))),
            ),
        ) {
            for x in 0..width {
                let mu1 = row_m1[x];
                let mu2 = row_m2[x];
                let mu11 = mu1 * mu1;
                let mu22 = mu2 * mu2;
                let mu12 = mu1 * mu2;
                let num_m = 1.0f32 - (mu1 - mu2).powi(2);
                let num_s = 2f32.mul_add(row_s12[x] - mu12, C2);
                let denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + C2;
                let mut d = 1.0f64 - f64::from((num_m * num_s) / denom_s);
                d = d.max(0.0);
                sum1[0] += d;
                sum1[1] += d.powi(4);
            }
        }
        plane_averages[c * 2] = one_per_pixels * sum1[0];
        plane_averages[c * 2 + 1] = (one_per_pixels * sum1[1]).sqrt().sqrt();
    }

    plane_averages
}

fn edge_diff_map(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    for c in 0..3 {
        let mut sum1 = [0.0f64; 4];
        for (row1, (row2, (rowm1, rowm2))) in img1[c].chunks_exact(width).zip(
            img2[c]
                .chunks_exact(width)
                .zip(mu1[c].chunks_exact(width).zip(mu2[c].chunks_exact(width))),
        ) {
            for x in 0..width {
                let d1: f64 = (1.0 + f64::from((row2[x] - rowm2[x]).abs()))
                    / (1.0 + f64::from((row1[x] - rowm1[x]).abs()))
                    - 1.0;
                // d1 > 0: distorted has an edge where original is smooth
                //         (indicating ringing, color banding, blockiness, etc)
                // d1 < 0: original has an edge where distorted is smooth
                //         (indicating smoothing, blurring, smearing, etc)
                let artifact = d1.max(0.0);
                sum1[0] += artifact;
                sum1[1] += artifact.powi(4);
                let detail_lost = (-d1).max(0.0);
                sum1[2] += detail_lost;
                sum1[3] += detail_lost.powi(4);
            }
        }
        plane_averages[c * 4] = one_per_pixels * sum1[0];
        plane_averages[c * 4 + 1] = (one_per_pixels * sum1[1]).sqrt().sqrt();
        plane_averages[c * 4 + 2] = one_per_pixels * sum1[2];
        plane_averages[c * 4 + 3] = (one_per_pixels * sum1[3]).sqrt().sqrt();
    }

    plane_averages
}

#[derive(Debug, Clone, Default)]
struct Msssim {
    pub scales: Vec<MsssimScale>,
}

#[derive(Debug, Clone, Copy, Default)]
struct MsssimScale {
    pub avg_ssim: [f64; 3 * 2],
    pub avg_edgediff: [f64; 3 * 4],
}

impl Msssim {
    #[allow(clippy::too_many_lines)]
    pub fn score(&self) -> f64 {
        const WEIGHT: [f64; 108] = [
            0.0_f64,
            0.0_f64,
            0.0_f64,
            1.003_547_935_251_235_3_f64,
            0.000_113_220_611_104_747_35_f64,
            0.000_404_429_918_236_859_36_f64,
            0.001_895_383_410_578_377_3_f64,
            0.0_f64,
            0.0_f64,
            8.982_542_997_575_905_f64,
            0.989_978_579_604_555_6_f64,
            0.0_f64,
            0.974_831_513_120_794_2_f64,
            0.958_157_516_993_797_3_f64,
            0.0_f64,
            0.513_361_177_795_294_6_f64,
            1.042_318_931_733_124_3_f64,
            0.000_308_010_928_520_841_f64,
            12.149_584_966_240_063_f64,
            0.956_557_724_811_546_7_f64,
            0.0_f64,
            1.040_666_812_313_682_4_f64,
            81.511_390_460_573_62_f64,
            0.305_933_918_953_309_46_f64,
            1.075_221_443_362_677_9_f64,
            1.103_904_236_946_461_1_f64,
            0.0_f64,
            1.021_911_638_819_618_f64,
            1.114_182_329_685_572_2_f64,
            0.973_084_575_144_170_5_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.983_391_842_609_550_5_f64,
            0.792_038_513_705_986_7_f64,
            0.971_074_041_151_405_3_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.538_707_790_315_263_8_f64,
            0.0_f64,
            3.403_694_560_115_580_4_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            2.337_569_295_661_117_f64,
            0.0_f64,
            5.707_946_510_901_609_f64,
            37.830_864_238_781_57_f64,
            0.0_f64,
            0.0_f64,
            3.825_820_059_430_518_5_f64,
            0.0_f64,
            0.0_f64,
            24.073_659_674_271_497_f64,
            0.0_f64,
            0.0_f64,
            13.181_871_265_286_068_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            10.007_501_212_628_95_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            52.514_283_856_038_91_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.994_646_426_789_441_7_f64,
            0.0_f64,
            0.0_f64,
            0.000_604_044_771_593_481_6_f64,
            0.0_f64,
            0.0_f64,
            0.994_517_149_137_407_2_f64,
            0.0_f64,
            2.826_004_380_945_437_6_f64,
            1.005_264_276_653_451_6_f64,
            8.201_441_997_546_244e-5_f64,
            12.154_041_855_876_695_f64,
            32.292_928_706_201_266_f64,
            0.992_837_130_387_521_f64,
            0.0_f64,
            30.719_255_178_446_03_f64,
            0.000_123_099_070_222_787_43_f64,
            0.0_f64,
            0.982_626_023_705_173_4_f64,
            0.0_f64,
            0.0_f64,
            0.998_092_836_783_765_1_f64,
            0.012_142_430_067_163_312_f64,
        ];

        let mut ssim = 0.0f64;

        let mut i = 0usize;
        for c in 0..3 {
            for scale in &self.scales {
                for n in 0..2 {
                    ssim += WEIGHT[i] * scale.avg_ssim[c * 2 + n].abs();
                    i += 1;
                    ssim += WEIGHT[i] * scale.avg_edgediff[c * 4 + n].abs();
                    i += 1;
                    ssim += WEIGHT[i] * scale.avg_edgediff[c * 4 + n + 2].abs();
                    i += 1;
                }
            }
        }

        ssim = ssim * 17.829_717_797_575_952_f64 - 1.634_169_143_917_183_f64;

        if ssim > 0.0f64 {
            ssim = 100.0f64 - 10.0f64 * ssim.powf(0.545_326_100_951_021_3_f64);
        } else {
            ssim = 100.0f64;
        }

        ssim
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use yuvxyb::Rgb;

    #[test]
    fn test_ssimulacra2() {
        let source = image::open(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("tank_source.png"),
        )
        .unwrap();
        let distorted = image::open(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("tank_distorted.png"),
        )
        .unwrap();
        let source_data = source
            .to_rgb32f()
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let source_data = Xyb::try_from(
            Rgb::new(
                source_data,
                source.width() as usize,
                source.height() as usize,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap(),
        )
        .unwrap();
        let distorted_data = distorted
            .to_rgb32f()
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let distorted_data = Xyb::try_from(
            Rgb::new(
                distorted_data,
                distorted.width() as usize,
                distorted.height() as usize,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap(),
        )
        .unwrap();
        let result = compute_frame_ssimulacra2(source_data, distorted_data).unwrap();
        let expected = 8.764_571_f64;
        assert!(
            (result - expected).abs() < 0.01f64,
            "Result {:.6} not equal to expected {:.6}",
            result,
            expected
        );
    }
}
