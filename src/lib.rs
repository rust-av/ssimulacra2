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

pub use blur::Blur;
pub use yuvxyb::{CastFromPrimitive, Frame, LinearRgb, Pixel, Plane, Rgb, Xyb, Yuv};
pub use yuvxyb::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic, YuvConfig};

// How often to downscale and score the input images.
// Each scaling step will downscale by a factor of two.
const NUM_SCALES: usize = 6;

/// Errors which can occur when attempting to calculate a SSIMULACRA2 score from two input images.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Ssimulacra2Error {
    /// The conversion from input image to [LinearRgb] (via [TryFrom]) returned an [Err].
    /// Note that the conversion from LinearRgb to [Xyb] cannot fail, which means that
    /// this is the only point of failure regarding image conversion.
    #[error("Failed to convert input image to linear RGB")]
    LinearRgbConversionFailed,

    /// The two input images do not have the same width and height.
    #[error("Source and distorted image width and height must be equal")]
    NonMatchingImageDimensions,

    /// One of the input images has a width and/or height of less than 8 pixels.
    /// This is not currently supported by the SSIMULACRA2 metric.
    #[error("Images must be at least 8x8 pixels")]
    InvalidImageSize,
}

/// Computes the SSIMULACRA2 score for a given input frame and the distorted
/// version of that frame.
///
/// # Errors
/// - If the source and distorted image width and height do not match
/// - If the source or distorted image cannot be converted to XYB successfully
/// - If the image is smaller than 8x8 pixels
pub fn compute_frame_ssimulacra2<T, U>(source: T, distorted: U) -> Result<f64, Ssimulacra2Error>
where
    LinearRgb: TryFrom<T> + TryFrom<U>,
{
    let Ok(mut img1) = LinearRgb::try_from(source) else {
        return Err(Ssimulacra2Error::LinearRgbConversionFailed);
    };

    let Ok(mut img2) = LinearRgb::try_from(distorted) else {
        return Err(Ssimulacra2Error::LinearRgbConversionFailed);
    };

    if img1.width() != img2.width() || img1.height() != img2.height() {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }

    if img1.width() < 8 || img1.height() < 8 {
        return Err(Ssimulacra2Error::InvalidImageSize);
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

        let mut img1 = Xyb::from(img1.clone());
        let mut img2 = Xyb::from(img2.clone());

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

// Get all components in more or less 0..1 range
// Range of Rec2020 with these adjustments:
//  X: 0.017223..0.998838
//  Y: 0.010000..0.855303
//  B: 0.048759..0.989551
// Range of sRGB:
//  X: 0.204594..0.813402
//  Y: 0.010000..0.855308
//  B: 0.272295..0.938012
// The maximum pixel-wise difference has to be <= 1 for the ssim formula to make
// sense.
fn make_positive_xyb(xyb: &mut Xyb) {
    for pix in xyb.data_mut().iter_mut() {
        pix[2] = (pix[2] - pix[1]) + 0.55;
        pix[0] = (pix[0]).mul_add(14.0, 0.42);
        pix[1] += 0.01;
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
                let mu_diff = mu1 - mu2;

                // Correction applied compared to the original SSIM formula, which has:
                //   luma_err = 2 * mu1 * mu2 / (mu1^2 + mu2^2)
                //            = 1 - (mu1 - mu2)^2 / (mu1^2 + mu2^2)
                // The denominator causes error in the darks (low mu1 and mu2) to weigh
                // more than error in the brights (high mu1 and mu2). This would make
                // sense if values correspond to linear luma. However, the actual values
                // are either gamma-compressed luma (which supposedly is already
                // perceptually uniform) or chroma (where weighing green more than red
                // or blue more than yellow does not make any sense at all). So it is
                // better to simply drop this denominator.
                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
                let num_s = 2f32.mul_add(row_s12[x] - mu12, C2);
                let denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + C2;
                // Use 1 - SSIM' so it becomes an error score instead of a quality
                // index. This makes it make sense to compute an L_4 norm.
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
                let artifact = d1.max(0.0);
                sum1[0] += artifact;
                sum1[1] += artifact.powi(4);

                // d1 < 0: original has an edge where distorted is smooth
                //         (indicating smoothing, blurring, smearing, etc)
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
    // The final score is based on a weighted sum of 108 sub-scores:
    // - for 6 scales (1:1 to 1:32)
    // - for 6 scales (1:1 to 1:32, downsampled in linear RGB)
    // - for 3 components (X + 0.5, Y, B - Y + 1.0)
    // - for 3 components (X, Y, B-Y, rescaled to 0..1 range)
    // - using 2 norms (the 1-norm and the 4-norm)
    // - using 2 norms (the 1-norm and the 4-norm)
    // - over 3 error maps:
    // - over 3 error maps:
    //     - SSIM
    //     - SSIM' (SSIM without the spurious gamma correction term)
    //     - "ringing" (distorted edges where there are no orig edges)
    //     - "ringing" (distorted edges where there are no orig edges)
    //     - "blurring" (orig edges where there are no distorted edges)
    //     - "blurring" (orig edges where there are no distorted edges)
    // The weights were obtained by running Nelder-Mead simplex search,
    // The weights were obtained by running Nelder-Mead simplex search,
    // optimizing to minimize MSE and maximize Kendall and Pearson correlation
    // optimizing to minimize MSE for the CID22 training set and to
    // for training data consisting of 17611 subjective quality scores,
    // maximize Kendall rank correlation (and with a lower weight,
    // validated on separate validation data consisting of 4292 scores.
    // also Pearson correlation) with the CID22 training set and the
    // TID2013, Kadid10k and KonFiG-IQA datasets.
    // Validation was done on the CID22 validation set.
    // Final results after tuning (Kendall | Spearman | Pearson):
    //    CID22:     0.6903 | 0.8805 | 0.8583
    //    TID2013:   0.6590 | 0.8445 | 0.8471
    //    KADID-10k: 0.6175 | 0.8133 | 0.8030
    //    KonFiG(F): 0.7668 | 0.9194 | 0.9136
    #[allow(clippy::too_many_lines)]
    pub fn score(&self) -> f64 {
        const WEIGHT: [f64; 108] = [
            0.0,
            0.000_737_660_670_740_658_6,
            0.0,
            0.0,
            0.000_779_348_168_286_730_9,
            0.0,
            0.0,
            0.000_437_115_573_010_737_9,
            0.0,
            1.104_172_642_665_734_6,
            0.000_662_848_341_292_71,
            0.000_152_316_327_837_187_52,
            0.0,
            0.001_640_643_745_659_975_4,
            0.0,
            1.842_245_552_053_929_8,
            11.441_172_603_757_666,
            0.0,
            0.000_798_910_943_601_516_3,
            0.000_176_816_438_078_653,
            0.0,
            1.878_759_497_954_638_7,
            10.949_069_906_051_42,
            0.0,
            0.000_728_934_699_150_807_2,
            0.967_793_708_062_683_3,
            0.0,
            0.000_140_034_242_854_358_84,
            0.998_176_697_785_496_7,
            0.000_319_497_559_344_350_53,
            0.000_455_099_211_379_206_3,
            0.0,
            0.0,
            0.001_364_876_616_324_339_8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            7.466_890_328_078_848,
            0.0,
            17.445_833_984_131_262,
            0.000_623_560_163_404_146_6,
            0.0,
            0.0,
            6.683_678_146_179_332,
            0.000_377_244_079_796_112_96,
            1.027_889_937_768_264,
            225.205_153_008_492_74,
            0.0,
            0.0,
            19.213_238_186_143_016,
            0.001_140_152_458_661_836_1,
            0.001_237_755_635_509_985,
            176.393_175_984_506_94,
            0.0,
            0.0,
            24.433_009_998_704_76,
            0.285_208_026_121_177_57,
            0.000_448_543_692_383_340_8,
            0.0,
            0.0,
            0.0,
            34.779_063_444_837_72,
            44.835_625_328_877_896,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000_868_055_657_329_169_8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000_531_319_187_435_874_7,
            0.0,
            0.000_165_338_141_613_791_12,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000_417_917_180_325_133_6,
            0.001_729_082_823_472_283_3,
            0.0,
            0.002_082_700_584_663_643_7,
            0.0,
            0.0,
            8.826_982_764_996_862,
            23.192_433_439_989_26,
            0.0,
            95.108_049_881_108_6,
            0.986_397_803_440_068_2,
            0.983_438_279_246_535_3,
            0.001_228_640_504_827_849_3,
            171.266_725_589_730_7,
            0.980_785_887_243_537_9,
            0.0,
            0.0,
            0.0,
            0.000_513_006_458_899_067_9,
            0.0,
            0.000_108_540_578_584_115_37,
        ];

        let mut ssim = 0.0f64;

        let mut i = 0usize;
        for c in 0..3 {
            for scale in &self.scales {
                for n in 0..2 {
                    ssim = WEIGHT[i].mul_add(scale.avg_ssim[c * 2 + n].abs(), ssim);
                    i += 1;
                    ssim = WEIGHT[i].mul_add(scale.avg_edgediff[c * 4 + n].abs(), ssim);
                    i += 1;
                    ssim = WEIGHT[i].mul_add(scale.avg_edgediff[c * 4 + n + 2].abs(), ssim);
                    i += 1;
                }
            }
        }

        ssim *= 0.956_238_261_683_484_4_f64;
        ssim = (6.248_496_625_763_138e-5 * ssim * ssim).mul_add(
            ssim,
            2.326_765_642_916_932f64.mul_add(ssim, -0.020_884_521_182_843_837 * ssim * ssim),
        );

        if ssim > 0.0f64 {
            ssim = ssim
                .powf(0.627_633_646_783_138_7)
                .mul_add(-10.0f64, 100.0f64);
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
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("tank_source.png"),
        )
        .unwrap();
        let distorted = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
        let expected = 17.398_505_f64;
        assert!(
            // SOMETHING is WEIRD with Github CI where it gives different results across DIFFERENT
            // RUNS
            (result - expected).abs() < 0.25f64,
            "Result {result:.6} not equal to expected {expected:.6}",
        );
    }
}
