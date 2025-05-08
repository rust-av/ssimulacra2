mod blur;

pub use blur::{Blur, BlurOperator};
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

    #[error("Error in gaussian_blur_f32")]
    GaussianBlurError,

    #[error("Error in fast_image_resize")]
    ResizeError,
}

/// Computes the SSIMULACRA2 score for a given input frame and the distorted
/// version of that frame.
///
/// # Errors
/// - If the source and distorted image width and height do not match
/// - If the source or distorted image cannot be converted to XYB successfully
/// - If the image is smaller than 8x8 pixels
#[inline(always)]
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

    let initial_size = width * height;
    let mut mul = [
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
    ];

    let mut sigma1_sq = [
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
    ];
    let mut sigma2_sq = [
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
    ];
    let mut sigma12 = [
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
    ];
    let mut mu1 = [
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
    ];
    let mut mu2 = [
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
        vec![0.0f32; initial_size],
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

            for c in &mut mul {
                c.truncate(width * height);
            }
            for c in &mut sigma1_sq {
                c.truncate(width * height);
            }
            for c in &mut sigma2_sq {
                c.truncate(width * height);
            }
            for c in &mut sigma12 {
                c.truncate(width * height);
            }
            for c in &mut mu1 {
                c.truncate(width * height);
            }
            for c in &mut mu2 {
                c.truncate(width * height);
            }

            blur.shrink_to(width, height);
        }

        // "Convert to XYB format"
        let mut img1_xyb = Xyb::from(img1.clone());
        let mut img2_xyb = Xyb::from(img2.clone());

        make_positive_xyb(&mut img1_xyb);
        make_positive_xyb(&mut img2_xyb);

        // "Convert to planar format"
        let img1_planar = xyb_to_planar(&img1_xyb);
        let img2_planar = xyb_to_planar(&img2_xyb);

        image_multiply(&img1_planar, &img1_planar, &mut mul);
        blur.blur(&mul, &mut sigma1_sq)?;

        image_multiply(&img2_planar, &img2_planar, &mut mul);
        blur.blur(&mul, &mut sigma2_sq)?;

        image_multiply(&img1_planar, &img2_planar, &mut mul);
        blur.blur(&mul, &mut sigma12)?;

        blur.blur(&img1_planar, &mut mu1)?;
        blur.blur(&img2_planar, &mut mu2)?;

        let avg_ssim = ssim_map(width, height, &mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12);
        let avg_edgediff = edge_diff_map(width, height, &img1_planar, &mu1, &img2_planar, &mu2);

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
#[inline(always)]
fn make_positive_xyb(xyb: &mut Xyb) {
    for pix in xyb.data_mut().iter_mut() {
        pix[2] = (pix[2] - pix[1]) + 0.55;
        pix[0] = (pix[0]).mul_add(14.0, 0.42);
        pix[1] += 0.01;
    }
}

#[inline(always)]
pub fn xyb_to_planar(xyb: &Xyb) -> [Vec<f32>; 3] {
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

#[inline(always)]
// 2.3625ms
pub fn image_multiply(img1: &[Vec<f32>; 3], img2: &[Vec<f32>; 3], out: &mut [Vec<f32>; 3]) {
    for ((plane1, plane2), out_plane) in img1.iter().zip(img2.iter()).zip(out.iter_mut()) {
        for ((&p1, &p2), o) in plane1.iter().zip(plane2.iter()).zip(out_plane.iter_mut()) {
            *o = p1 * p2;
        }
    }
}

#[inline(always)]
/// 1.7866ms
pub fn downscale_by_2(in_data: &LinearRgb) -> LinearRgb {
    const SCALE: usize = 2;
    let in_w = in_data.width();
    let in_h = in_data.height();
    let out_w = (in_w + SCALE - 1) / SCALE;
    let out_h = (in_h + SCALE - 1) / SCALE;
    let mut out_data = vec![[0.0f32; 3]; out_w * out_h];
    let normalize = 1f32 / (SCALE * SCALE) as f32;
    let in_data = in_data.data();

    // "Process the inner area without boundaries first"
    let safe_h = in_h - (in_h % SCALE);
    let safe_w = in_w - (in_w % SCALE);

    for oy in 0..(safe_h / SCALE) {
        let y_base = oy * SCALE;
        for ox in 0..(safe_w / SCALE) {
            let x_base = ox * SCALE;
            let out_idx = oy * out_w + ox;

            // "Unroll the loop to access all pixels directly"
            let p00 = in_data[y_base * in_w + x_base];
            let p01 = in_data[y_base * in_w + (x_base + 1)];
            let p10 = in_data[(y_base + 1) * in_w + x_base];
            let p11 = in_data[(y_base + 1) * in_w + (x_base + 1)];

            out_data[out_idx][0] = (p00[0] + p01[0] + p10[0] + p11[0]) * normalize;
            out_data[out_idx][1] = (p00[1] + p01[1] + p10[1] + p11[1]) * normalize;
            out_data[out_idx][2] = (p00[2] + p01[2] + p10[2] + p11[2]) * normalize;
        }
    }

    if safe_w < in_w || safe_h < in_h {
        for oy in 0..out_h {
            let y_start = oy * SCALE;
            for ox in 0..out_w {
                if oy < safe_h / SCALE && ox < safe_w / SCALE {
                    continue;
                }

                let x_start = ox * SCALE;
                let out_idx = oy * out_w + ox;
                let mut r = 0.0;
                let mut g = 0.0;
                let mut b = 0.0;
                let mut count = 0;

                for dy in 0..SCALE {
                    let y = y_start + dy;
                    if y >= in_h {
                        continue;
                    }

                    for dx in 0..SCALE {
                        let x = x_start + dx;
                        if x >= in_w {
                            continue;
                        }

                        let pixel = in_data[y * in_w + x];
                        r += pixel[0];
                        g += pixel[1];
                        b += pixel[2];
                        count += 1;
                    }
                }

                let scale = if count > 0 { 1.0 / count as f32 } else { 0.0 };
                out_data[out_idx][0] = r * scale;
                out_data[out_idx][1] = g * scale;
                out_data[out_idx][2] = b * scale;
            }
        }
    }

    LinearRgb::new(out_data, out_w, out_h).expect("Resolution and data size match")
}

#[inline(always)]
// 4.5360ms
pub fn ssim_map(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let results: Vec<(usize, [f64; 2])> = (0..3)
            .into_par_iter()
            .map(|c| {
                let sums = compute_channel_sums(width, c, m1, m2, s11, s22, s12);
                (c, sums)
            })
            .collect();

        for (c, sums) in results {
            plane_averages[c * 2] = one_per_pixels * sums[0];
            plane_averages[c * 2 + 1] = (one_per_pixels * sums[1]).sqrt().sqrt();
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        for c in 0..3 {
            let sums = compute_channel_sums(width, c, m1, m2, s11, s22, s12);
            plane_averages[c * 2] = one_per_pixels * sums[0];
            plane_averages[c * 2 + 1] = (one_per_pixels * sums[1]).sqrt().sqrt();
        }
    }

    plane_averages
}

#[inline(always)]
fn compute_channel_sums(
    width: usize,
    c: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 2] {
    const C2: f32 = 0.0009f32;
    let mut sums = [0.0f64; 2];

    for (row_idx, (row_m1, row_m2)) in m1[c]
        .chunks_exact(width)
        .zip(m2[c].chunks_exact(width))
        .enumerate()
    {
        let row_offset = row_idx * width;
        for x in 0..width {
            let mu1 = row_m1[x];
            let mu2 = row_m2[x];
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
            let num_m = mu_diff.mul_add(-mu_diff, 1.0);
            let mu12 = mu1 * mu2;
            let sigma12 = s12[c][row_offset + x] - mu12;
            let num_s = 2.0f32.mul_add(sigma12, C2);
            let sigma1_sq = s11[c][row_offset + x] - mu1 * mu1;
            let sigma2_sq = s22[c][row_offset + x] - mu2 * mu2;
            let denom_s = sigma1_sq + sigma2_sq + C2;

            let ssim = (num_m * num_s) / denom_s;
            let err = (1.0 - f64::from(ssim)).max(0.0);

            sums[0] += err;
            let err_sq = err * err;
            sums[1] += err_sq * err_sq;
        }
    }

    sums
}

#[inline(always)]
// 12.492ms ssimulacra2 bench
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
        // cache
        let mut artifact_sum = 0.0f64;
        let mut artifact_pow4_sum = 0.0f64;
        let mut detail_lost_sum = 0.0f64;
        let mut detail_lost_pow4_sum = 0.0f64;

        for (row1, (row2, (rowm1, rowm2))) in img1[c].chunks_exact(width).zip(
            img2[c]
                .chunks_exact(width)
                .zip(mu1[c].chunks_exact(width).zip(mu2[c].chunks_exact(width))),
        ) {
            for x in 0..width {
                let diff1 = f64::from((row1[x] - rowm1[x]).abs());
                let diff2 = f64::from((row2[x] - rowm2[x]).abs());

                let d1 = (1.0 + diff2) / (1.0 + diff1) - 1.0;

                // d1 > 0: distorted has an edge where original is smooth
                //         (indicating ringing, color banding, blockiness, etc)
                let artifact = 0.5 * (d1 + d1.abs()); // if d1 is positive, d1; if negative, 0
                artifact_sum += artifact;

                let artifact_squared = artifact * artifact;
                artifact_pow4_sum += artifact_squared * artifact_squared;

                // d1 < 0: original has an edge where distorted is smooth
                //         (indicating smoothing, blurring, smearing, etc)
                let detail_lost = 0.5 * (-d1 + d1.abs()); // if d1 is negative, -d1; if positive, 0
                detail_lost_sum += detail_lost;
                let detail_lost_squared = detail_lost * detail_lost;
                detail_lost_pow4_sum += detail_lost_squared * detail_lost_squared;
            }
        }

        // Calculate the averages for the current channel
        let base_idx = c * 4;
        plane_averages[base_idx] = one_per_pixels * artifact_sum;
        plane_averages[base_idx + 1] = (one_per_pixels * artifact_pow4_sum).sqrt().sqrt();
        plane_averages[base_idx + 2] = one_per_pixels * detail_lost_sum;
        plane_averages[base_idx + 3] = (one_per_pixels * detail_lost_pow4_sum).sqrt().sqrt();
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
    #[inline(always)]
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
    use std::{path::PathBuf, time::Instant};

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
        let start = Instant::now();
        let result = compute_frame_ssimulacra2(source_data, distorted_data).unwrap();
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);
        let expected = 17.398_505_f64;
        println!("Result: {result:.6}");
        println!("Expected: {expected:.6}");
        assert!(
            // SOMETHING is WEIRD with Github CI where it gives different results across DIFFERENT
            // RUNS
            (result - expected).abs() < 0.25f64,
            "Result {result:.6} not equal to expected {expected:.6}",
        );
    }

    #[test]
    fn test2_ssimulacra2() {
        let source = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("test_image_4.png"),
        )
        .unwrap();
        let distorted = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("test_image_4.jpg"),
        )
        .unwrap();

        println!("source width: {}", source.width());
        println!("source height: {}", source.height());
        println!("distorted width: {}", distorted.width());
        println!("distorted height: {}", distorted.height());

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
        let start = Instant::now();
        let result = compute_frame_ssimulacra2(source_data, distorted_data).unwrap();
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);
        let expected = 84.031_931_f64;

        println!("Result: {result:.6}");
        println!("Expected: {expected:.6}");
        assert!(
            // SOMETHING is WEIRD with Github CI where it gives different results across DIFFERENT
            // RUNS
            (result - expected).abs() < 1.0f64,
            "Result {result:.6} not equal to expected {expected:.6}",
        );
    }
}
