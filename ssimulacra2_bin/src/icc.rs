use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use lcms2::{ColorSpaceSignature, Intent, PixelFormat, Profile, Transform};

pub struct DecodedImage {
    pub pixels: Vec<[f32; 3]>,
    pub width: usize,
    pub height: usize,
    pub alpha: Option<Vec<f32>>,
}

pub fn decode_image(path: &Path, use_icc: bool) -> DecodedImage {
    let ext = path
        .extension()
        .map(|e| e.to_ascii_lowercase().to_string_lossy().into_owned())
        .unwrap_or_default();

    match ext.as_str() {
        "png" => decode_png(path, use_icc),
        "jpg" | "jpeg" => decode_jpeg(path, use_icc),
        _ => decode_fallback(path),
    }
}

fn decode_png(path: &Path, use_icc: bool) -> DecodedImage {
    let file = File::open(path).expect("Failed to open PNG file");
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().expect("Failed to read PNG info");

    let info = reader.info();
    let width = info.width as usize;
    let height = info.height as usize;
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;
    let icc_profile = if use_icc {
        info.icc_profile.as_ref().map(|cow| cow.to_vec())
    } else {
        None
    };

    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .expect("Decoded image too large for system RAM")
    ];
    let output_info = reader
        .next_frame(&mut buf)
        .expect("Failed to decode PNG frame");
    let raw = &buf[..output_info.buffer_size()];

    let has_alpha = matches!(
        color_type,
        png::ColorType::Rgba | png::ColorType::GrayscaleAlpha
    );
    let is_grayscale = matches!(
        color_type,
        png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha
    );
    let is_16bit = bit_depth == png::BitDepth::Sixteen;

    if is_16bit {
        decode_png_16bit(raw, width, height, has_alpha, is_grayscale, &icc_profile)
    } else {
        decode_png_8bit(raw, width, height, has_alpha, is_grayscale, &icc_profile)
    }
}

fn decode_png_8bit(
    raw: &[u8],
    width: usize,
    height: usize,
    has_alpha: bool,
    is_grayscale: bool,
    icc_profile: &Option<Vec<u8>>,
) -> DecodedImage {
    let pixel_count = width * height;
    let channels = if has_alpha {
        if is_grayscale { 2 } else { 4 }
    } else if is_grayscale {
        1
    } else {
        3
    };

    assert_eq!(raw.len(), pixel_count * channels);

    let (rgb_bytes, alpha) =
        extract_rgb_alpha_u8(raw, pixel_count, channels, has_alpha, is_grayscale);

    let rgb_bytes = apply_icc_transform_u8(rgb_bytes, icc_profile);

    let pixels: Vec<[f32; 3]> = rgb_bytes
        .chunks_exact(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();

    DecodedImage {
        pixels,
        width,
        height,
        alpha,
    }
}

fn decode_png_16bit(
    raw: &[u8],
    width: usize,
    height: usize,
    has_alpha: bool,
    is_grayscale: bool,
    icc_profile: &Option<Vec<u8>>,
) -> DecodedImage {
    let pixel_count = width * height;
    let channels = if has_alpha {
        if is_grayscale { 2 } else { 4 }
    } else if is_grayscale {
        1
    } else {
        3
    };

    assert_eq!(raw.len(), pixel_count * channels * 2);

    let (rgb_u16, alpha) =
        extract_rgb_alpha_u16(raw, pixel_count, channels, has_alpha, is_grayscale);

    let rgb_u16 = apply_icc_transform_u16(rgb_u16, icc_profile);

    let pixels: Vec<[f32; 3]> = rgb_u16
        .chunks_exact(3)
        .map(|c| {
            [
                c[0] as f32 / 65535.0,
                c[1] as f32 / 65535.0,
                c[2] as f32 / 65535.0,
            ]
        })
        .collect();

    DecodedImage {
        pixels,
        width,
        height,
        alpha,
    }
}

fn extract_rgb_alpha_u8(
    raw: &[u8],
    pixel_count: usize,
    channels: usize,
    has_alpha: bool,
    is_grayscale: bool,
) -> (Vec<u8>, Option<Vec<f32>>) {
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    let mut alpha = if has_alpha {
        Some(Vec::with_capacity(pixel_count))
    } else {
        None
    };

    for i in 0..pixel_count {
        let base = i * channels;
        if is_grayscale {
            let g = raw[base];
            rgb.push(g);
            rgb.push(g);
            rgb.push(g);
            if has_alpha {
                alpha.as_mut().unwrap().push(raw[base + 1] as f32 / 255.0);
            }
        } else {
            rgb.push(raw[base]);
            rgb.push(raw[base + 1]);
            rgb.push(raw[base + 2]);
            if has_alpha {
                alpha.as_mut().unwrap().push(raw[base + 3] as f32 / 255.0);
            }
        }
    }

    (rgb, alpha)
}

fn extract_rgb_alpha_u16(
    raw: &[u8],
    pixel_count: usize,
    channels: usize,
    has_alpha: bool,
    is_grayscale: bool,
) -> (Vec<u16>, Option<Vec<f32>>) {
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    let mut alpha = if has_alpha {
        Some(Vec::with_capacity(pixel_count))
    } else {
        None
    };

    for i in 0..pixel_count {
        let base = i * channels * 2;
        if is_grayscale {
            let g = u16::from_be_bytes([raw[base], raw[base + 1]]);
            rgb.push(g);
            rgb.push(g);
            rgb.push(g);
            if has_alpha {
                let a = u16::from_be_bytes([raw[base + 2], raw[base + 3]]);
                alpha.as_mut().unwrap().push(a as f32 / 65535.0);
            }
        } else {
            rgb.push(u16::from_be_bytes([raw[base], raw[base + 1]]));
            rgb.push(u16::from_be_bytes([raw[base + 2], raw[base + 3]]));
            rgb.push(u16::from_be_bytes([raw[base + 4], raw[base + 5]]));
            if has_alpha {
                let a = u16::from_be_bytes([raw[base + 6], raw[base + 7]]);
                alpha.as_mut().unwrap().push(a as f32 / 65535.0);
            }
        }
    }

    (rgb, alpha)
}

fn apply_icc_transform_u8(rgb: Vec<u8>, icc_profile: &Option<Vec<u8>>) -> Vec<u8> {
    let Some(icc_data) = icc_profile else {
        return rgb;
    };

    let src_profile = match Profile::new_icc(icc_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Warning: failed to parse ICC profile ({e}), assuming sRGB");
            return rgb;
        }
    };

    if src_profile.color_space() != ColorSpaceSignature::RgbData {
        eprintln!("Warning: embedded ICC profile is not an RGB profile, assuming sRGB");
        return rgb;
    }

    let dst_profile = Profile::new_srgb();

    let transform = match Transform::new(
        &src_profile,
        PixelFormat::RGB_8,
        &dst_profile,
        PixelFormat::RGB_8,
        Intent::RelativeColorimetric,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Warning: failed to create ICC transform ({e}), assuming sRGB");
            return rgb;
        }
    };

    let mut dst = vec![0u8; rgb.len()];
    transform.transform_pixels(&rgb, &mut dst);
    dst
}

fn apply_icc_transform_u16(rgb: Vec<u16>, icc_profile: &Option<Vec<u8>>) -> Vec<u16> {
    let Some(icc_data) = icc_profile else {
        return rgb;
    };

    let src_profile = match Profile::new_icc(icc_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Warning: failed to parse ICC profile ({e}), assuming sRGB");
            return rgb;
        }
    };

    if src_profile.color_space() != ColorSpaceSignature::RgbData {
        eprintln!("Warning: embedded ICC profile is not an RGB profile, assuming sRGB");
        return rgb;
    }

    let dst_profile = Profile::new_srgb();

    let transform = match Transform::new(
        &src_profile,
        PixelFormat::RGB_16,
        &dst_profile,
        PixelFormat::RGB_16,
        Intent::RelativeColorimetric,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Warning: failed to create ICC transform ({e}), assuming sRGB");
            return rgb;
        }
    };

    let mut dst = vec![0u16; rgb.len()];
    transform.transform_pixels(&rgb, &mut dst);
    dst
}

/// Naive CMYK -> RGB conversion, ignoring color management. Used when no usable
/// CMYK ICC profile is embedded. `cmyk` is interleaved C,M,Y,K (one byte each),
/// already un-inverted by the JPEG decoder.
fn naive_cmyk_to_rgb(cmyk: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in cmyk.chunks_exact(4) {
        let c = chunk[0] as f32 / 255.0;
        let m = chunk[1] as f32 / 255.0;
        let y = chunk[2] as f32 / 255.0;
        let k = chunk[3] as f32 / 255.0;
        rgb.push(((1.0 - c) * (1.0 - k) * 255.0) as u8);
        rgb.push(((1.0 - m) * (1.0 - k) * 255.0) as u8);
        rgb.push(((1.0 - y) * (1.0 - k) * 255.0) as u8);
    }
    rgb
}

/// Convert a 4-channel CMYK buffer to sRGB.
///
/// When the JPEG embeds a CMYK ICC profile we color-manage the conversion with
/// lcms2 (CMYK -> sRGB) instead of the naive formula, so the profile is honored
/// rather than dropped. We only do this when the profile's color space is
/// actually CMYK; a profile that claims any other space on CMYK data is treated
/// as unusable and we fall back to the naive conversion with a warning.
///
/// Note: jpeg-decoder 0.3.1 reports both true-CMYK and YCCK sources as
/// `PixelFormat::CMYK32` and gives no way to tell them apart. For YCCK-origin
/// files the buffer is not real CMYK, so color management (and the naive path)
/// can be off. That is a pre-existing decoder limitation, not something this
/// path can resolve without distinguishing the source transform.
fn decode_cmyk_to_rgb(cmyk: &[u8], pixel_count: usize, icc_profile: &Option<Vec<u8>>) -> Vec<u8> {
    let Some(icc_data) = icc_profile else {
        return naive_cmyk_to_rgb(cmyk, pixel_count);
    };

    let src_profile = match Profile::new_icc(icc_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Warning: failed to parse ICC profile ({e}), using naive CMYK conversion");
            return naive_cmyk_to_rgb(cmyk, pixel_count);
        }
    };

    if src_profile.color_space() != ColorSpaceSignature::CmykData {
        eprintln!(
            "Warning: embedded ICC profile is not a CMYK profile, using naive CMYK conversion"
        );
        return naive_cmyk_to_rgb(cmyk, pixel_count);
    }

    let dst_profile = Profile::new_srgb();

    let transform = match Transform::new(
        &src_profile,
        PixelFormat::CMYK_8,
        &dst_profile,
        PixelFormat::RGB_8,
        Intent::RelativeColorimetric,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "Warning: failed to create CMYK ICC transform ({e}), using naive CMYK conversion"
            );
            return naive_cmyk_to_rgb(cmyk, pixel_count);
        }
    };

    let mut dst = vec![0u8; pixel_count * 3];
    transform.transform_pixels(cmyk, &mut dst);
    dst
}

fn decode_jpeg(path: &Path, use_icc: bool) -> DecodedImage {
    let file = File::open(path).expect("Failed to open JPEG file");
    let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
    let raw = decoder.decode().expect("Failed to decode JPEG");
    let metadata = decoder.info().expect("Failed to get JPEG metadata");

    let width = metadata.width as usize;
    let height = metadata.height as usize;
    let pixel_count = width * height;

    let icc_profile = if use_icc { decoder.icc_profile() } else { None };

    let rgb_bytes = match metadata.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => apply_icc_transform_u8(raw, &icc_profile),
        jpeg_decoder::PixelFormat::L8 => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for &g in &raw {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            apply_icc_transform_u8(rgb, &icc_profile)
        }
        jpeg_decoder::PixelFormat::CMYK32 => decode_cmyk_to_rgb(&raw, pixel_count, &icc_profile),
        jpeg_decoder::PixelFormat::L16 => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for chunk in raw.chunks_exact(2) {
                let g = (u16::from_ne_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            apply_icc_transform_u8(rgb, &icc_profile)
        }
    };

    let pixels: Vec<[f32; 3]> = rgb_bytes
        .chunks_exact(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();

    DecodedImage {
        pixels,
        width,
        height,
        alpha: None,
    }
}

fn decode_fallback(path: &Path) -> DecodedImage {
    let img = image::open(path).expect("Failed to open image file");
    let width = img.width() as usize;
    let height = img.height() as usize;

    let pixels = img
        .to_rgb32f()
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();

    DecodedImage {
        pixels,
        width,
        height,
        alpha: None,
    }
}

pub fn composite_over_background(pixels: &[[f32; 3]], alpha: &[f32], bg: f32) -> Vec<[f32; 3]> {
    pixels
        .iter()
        .zip(alpha.iter())
        .map(|(px, &a)| {
            [
                px[0] * a + bg * (1.0 - a),
                px[1] * a + bg * (1.0 - a),
                px[2] * a + bg * (1.0 - a),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // One pure-cyan and one pure-black pixel (C,M,Y,K interleaved, un-inverted).
    const CMYK_PIXELS: [u8; 8] = [255, 0, 0, 0, 0, 0, 0, 255];

    #[test]
    fn naive_cmyk_matches_formula() {
        let rgb = naive_cmyk_to_rgb(&CMYK_PIXELS, 2);
        // Pure cyan: (1-1)(1-0)=0 R, (1-0)(1-0)=1 G/B -> [0,255,255].
        assert_eq!(&rgb[0..3], &[0, 255, 255]);
        // Pure black (K=1): every channel (1-x)(1-1)=0 -> [0,0,0].
        assert_eq!(&rgb[3..6], &[0, 0, 0]);
    }

    #[test]
    fn cmyk_without_profile_uses_naive() {
        let managed = decode_cmyk_to_rgb(&CMYK_PIXELS, 2, &None);
        let naive = naive_cmyk_to_rgb(&CMYK_PIXELS, 2);
        assert_eq!(managed, naive);
    }

    #[test]
    fn cmyk_with_non_cmyk_profile_falls_back_to_naive() {
        // An sRGB profile is an RGB profile, not CMYK. Feeding it as the
        // embedded profile for CMYK data must not be misapplied: we expect a
        // clean fall back to the naive conversion rather than garbage or a panic.
        let srgb_icc = Profile::new_srgb().icc().expect("serialize sRGB profile");
        let managed = decode_cmyk_to_rgb(&CMYK_PIXELS, 2, &Some(srgb_icc));
        let naive = naive_cmyk_to_rgb(&CMYK_PIXELS, 2);
        assert_eq!(
            managed, naive,
            "a non-CMYK profile on CMYK data should fall back to the naive conversion"
        );
    }

    #[test]
    fn rgb_transform_skips_non_rgb_profile() {
        // A non-RGB (here CMYK) profile must not be applied to RGB pixels;
        // apply_icc_transform_u8 should return the input unchanged.
        let cmyk_icc = Profile::new_srgb().icc().expect("serialize profile");
        // Build a genuinely-CMYK profile by round-tripping through color space.
        let mut p = Profile::new_icc(&cmyk_icc).expect("parse");
        p.set_color_space(ColorSpaceSignature::CmykData);
        let cmyk_icc = p.icc().expect("serialize cmyk-tagged profile");

        let rgb = vec![10u8, 20, 30, 40, 50, 60];
        let out = apply_icc_transform_u8(rgb.clone(), &Some(cmyk_icc));
        assert_eq!(out, rgb, "non-RGB profile must leave RGB pixels untouched");
    }
}
