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
pub use yuvxyb::{CastFromPrimitive, Frame, Pixel, Plane, Xyb, Yuv};

const NUM_SCALES: usize = 6;

/// Computes the SSIMULACRA2 score for a given input frame and the distorted
/// version of that frame.
///
/// # Errors
/// - If the source and distorted image width and height do not match
/// - If the source or distorted image cannot be converted to XYB successfully
/// - If the image is smaller than 8x8 pixels
pub fn compute_frame_ssimulacra2<T: Pixel>(source: &Yuv<T>, distorted: &Yuv<T>) -> Result<f64> {
    let mut img1: Xyb = source.try_into()?;
    let mut img2: Xyb = distorted.try_into()?;

    if img1.width() != img2.width() || img1.height() != img2.height() {
        bail!("Source and distorted image width and height must be equal");
    }

    if img1.width() < 8 || img1.height() < 8 {
        bail!("Images must be at least 8x8 pixels");
    }

    make_positive_xyb(&mut img1);
    make_positive_xyb(&mut img2);

    let mut width = img1.width();
    let mut height = img1.height();

    // SSIMULACRA2 works with the data in a planar format,
    // so we need to convert to that.
    let mut img1 = xyb_to_planar(&img1);
    let mut img2 = xyb_to_planar(&img2);

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
            (img1, _, _) = downscale_by_2(&img1, width, height);
            (img2, width, height) = downscale_by_2(&img2, width, height);
        }
        for c in &mut mul {
            c.truncate(width * height);
        }
        blur.shrink_to(width, height);

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
        pix[2] += 1.0 - pix[1];
        pix[0] += 0.5;
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

fn downscale_by_2(
    in_data: &[Vec<f32>; 3],
    in_w: usize,
    in_h: usize,
) -> ([Vec<f32>; 3], usize, usize) {
    const SCALE: usize = 2;
    let out_w = (in_w + SCALE - 1) / SCALE;
    let out_h = (in_h + SCALE - 1) / SCALE;
    let mut out_data = [
        vec![0.0f32; out_w * out_h],
        vec![0.0f32; out_w * out_h],
        vec![0.0f32; out_w * out_h],
    ];
    let normalize = 1f32 / (SCALE * SCALE) as f32;
    for c in 0..3 {
        let in_plane = &in_data[c];
        let out_plane = &mut out_data[c];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut sum = 0f32;
                for iy in 0..SCALE {
                    for ix in 0..SCALE {
                        let x = (ox * SCALE + ix).min(in_h - 1);
                        let y = (oy * SCALE + iy).min(in_h - 1);
                        let in_pix = in_plane[y * in_h + x];

                        sum += in_pix;
                    }
                }
                let out_pix = &mut out_plane[oy * out_w + ox];
                *out_pix = sum * normalize;
            }
        }
    }

    (out_data, out_w, out_h)
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
    const C1: f32 = 0.0001f32;
    const C2: f32 = 0.0003f32;

    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    for c in 0..3 {
        let mut sum1 = [0.0f64; 2];
        for y in 0..height {
            let row_m1 = &m1[c][(y * width)..][..width];
            let row_m2 = &m2[c][(y * width)..][..width];
            let row_s11 = &s11[c][(y * width)..][..width];
            let row_s22 = &s22[c][(y * width)..][..width];
            let row_s12 = &s12[c][(y * width)..][..width];
            for x in 0..width {
                let mu1 = row_m1[x];
                let mu2 = row_m2[x];
                let mu11 = mu1 * mu1;
                let mu22 = mu2 * mu2;
                let mu12 = mu1 * mu2;
                let num_m = 2f32.mul_add(mu12, C1);
                let num_s = 2f32.mul_add(row_s12[x] - mu12, C2);
                let denom_m = mu11 + mu22 + C1;
                let denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + C2;
                let mut d = 1.0f64 - f64::from((num_m * num_s) / (denom_m * denom_s));
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
        for y in 0..height {
            let row1 = &img1[c][(y * width)..][..width];
            let row2 = &img2[c][(y * width)..][..width];
            let rowm1 = &mu1[c][(y * width)..][..width];
            let rowm2 = &mu2[c][(y * width)..][..width];
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
                let detail_lost = -d1.max(0.0);
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
            4.219_667_647_997_749e-5f64,
            0.012_686_211_358_327_482f64,
            3.107_147_477_665_606e-5f64,
            0.000_543_596_238_167_687_3f64,
            0.093_951_297_338_375_15f64,
            0.000_231_164_895_018_842_74f64,
            3.116_178_275_375_247_6e-5f64,
            0.039_270_859_874_546_04f64,
            3.112_320_351_661_424e-5f64,
            15.207_946_778_270_552f64,
            0.116_853_730_606_454_32f64,
            0.108_258_830_426_009_81f64,
            3.116_785_767_387_498e-5f64,
            3.131_457_976_301_988e-5f64,
            3.114_664_519_432_431e-5f64,
            3.111_881_734_196_853e-5f64,
            0.150_526_079_086_462_2f64,
            1.181_932_253_296_347f64,
            0.023_779_401_135_092_804f64,
            3.118_721_767_259_025e-5f64,
            3.107_147_477_665_606e-5f64,
            3.107_147_477_665_606e-5f64,
            0.292_630_711_597_291_26f64,
            100.0f64,
            0.078_351_919_030_236_42f64,
            0.308_749_239_640_701f64,
            3.110_101_392_123_088e-5f64,
            0.033_134_729_290_677_18f64,
            1.261_558_573_839_896_7f64,
            1.286_504_153_416_386_1f64,
            0.000_500_715_801_872_941_8f64,
            3.114_135_552_706_454e-5f64,
            3.107_147_477_665_606e-5f64,
            0.099_621_988_698_567_2f64,
            0.074_444_825_774_388_82f64,
            0.113_724_270_846_116_47f64,
            5.518_066_533_005_683e-5f64,
            3.135_558_661_193_638e-5f64,
            3.116_549_250_103_961_6e-5f64,
            0.347_509_429_646_832_73f64,
            3.456_527_094_525_263_5e-5f64,
            4.088_543_972_599_083_5f64,
            3.401_042_790_207_587e-5f64,
            3.107_147_477_665_606e-5f64,
            3.131_677_581_003_078_4e-5f64,
            0.003_537_287_786_951_06f64,
            0.000_288_918_817_458_960_75f64,
            13.567_765_144_191_44f64,
            28.427_922_207_790_395f64,
            4.698_319_951_601_526e-5f64,
            3.124_776_402_918_527_7e-5f64,
            0.130_492_430_895_520_2f64,
            2.812_834_792_796_773_6f64,
            7.902_846_378_027_295e-5f64,
            1.310_663_427_102_324_8f64,
            0.000_215_730_430_846_994_28f64,
            0.000_130_161_602_971_856_64f64,
            3.406_144_249_596_765_8f64,
            4.460_412_915_533_889f64,
            3.107_147_477_665_606e-5f64,
            3.277_361_057_918_426_5e-5f64,
            0.103_694_572_772_048_52f64,
            3.629_363_118_118_345e-5f64,
            0.000_848_350_990_510_504_7f64,
            1.193_383_042_496_474_2f64,
            3.342_669_917_216_767e-5f64,
            3.112_936_463_123_272_5e-5f64,
            3.111_597_216_765_016e-5f64,
            0.002_772_786_993_656_906f64,
            5.506_805_306_998_43e-5f64,
            3.107_147_477_665_606e-5f64,
            3.113_120_547_104_664e-5f64,
            3.109_181_778_038_206e-5f64,
            3.107_147_477_665_606e-5f64,
            3.111_874_829_531_125e-5f64,
            3.271_770_143_775_665e-5f64,
            0.000_159_264_837_603_090_3f64,
            7.958_992_275_525_212e-5f64,
            3.276_592_137_968_492_6e-5f64,
            3.119_778_402_449_48e-5f64,
            3.117_375_426_220_37e-5f64,
            3.269_854_031_795_471_6e-5f64,
            0.000_206_695_229_672_426_7f64,
            8.396_345_538_652_65e-5f64,
            3.444_512_635_775_165_4e-5f64,
            4.973_593_015_122_901e-5f64,
            3.108_593_217_115_985e-5f64,
            7.448_916_645_891_313e-5f64,
            0.000_650_549_577_087_655_7f64,
            4.342_326_567_408_072_4e-5f64,
            7.247_563_231_427_279e-5f64,
            0.000_212_235_447_640_596_32f64,
            3.117_729_633_383_97e-5f64,
            0.406_753_628_973_467_8f64,
            0.138_980_498_370_882_55f64,
            4.541_178_136_114_84f64,
            0.068_534_911_051_404_75f64,
            0.155_812_526_556_593_17f64,
            0.099_826_649_210_247_64f64,
            3.440_168_932_652_795f64,
            0.128_296_531_034_086_23f64,
            56.599_309_867_339_67f64,
            5.773_410_728_426_853e-5f64,
            0.106_744_046_353_943_3f64,
            3.108_444_898_647_367e-5f64,
            3.374_827_724_533_791e-5f64,
            0.020_250_432_987_237_055f64,
            0.133_468_423_072_341_2f64,
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

        ssim = ssim * 11.480_665_013_024_748f64 - 1.020_461_049_104_017_4f64;

        if ssim > 0.0f64 {
            ssim = 100.0f64 - 10.0f64 * ssim.powf(0.640_203_200_929_897_9f64);
        } else {
            ssim = 100.0f64;
        }

        ssim
    }
}
