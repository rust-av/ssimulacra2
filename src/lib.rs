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

    let mut mul = vec![[0.0f32; 3]; img1.width() * img1.height()];
    let mut blur = Blur::new(img1.width(), img1.height());
    let mut msssim = Msssim::default();

    for scale in 0..NUM_SCALES {
        if img1.width() < 8 || img1.height() < 8 {
            break;
        }

        if scale > 0 {
            img1 = downscale_by_2(&img1);
            img2 = downscale_by_2(&img2);
        }
        mul.truncate(img1.width() * img1.height());
        blur.shrink_to(img1.width(), img1.height());

        image_multiply(&img1, &img1, &mut mul);
        let sigma1_sq = blur.blur(&mul);

        image_multiply(&img2, &img2, &mut mul);
        let sigma2_sq = blur.blur(&mul);

        image_multiply(&img1, &img2, &mut mul);
        let sigma12 = blur.blur(&mul);

        let mu1 = blur.blur(&img1);
        let mu2 = blur.blur(&img2);

        let avg_ssim = ssim_map(&mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12);
        let avg_edgediff = edge_diff_map(&img1, &mu1, &img2, &mu2);
        msssim.scales.push(MsssimScale {
            avg_ssim,
            avg_edgediff,
        });
    }

    Ok(msssim.score())
}

fn make_positive_xyb(xyb: &mut Xyb) {
    xyb.data_mut().iter_mut().for_each(|pix| {
        pix[2] += 1.0 - pix[1];
        pix[0] += 0.5;
    });
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
    pub fn score(&self) -> f64 {
        todo!()
    }
}
