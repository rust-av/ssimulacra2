use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use lcms2::{Intent, PixelFormat, Profile, Transform};

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

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let output_info = reader.next_frame(&mut buf).expect("Failed to decode PNG frame");
    let raw = &buf[..output_info.buffer_size()];

    let has_alpha = matches!(color_type, png::ColorType::Rgba | png::ColorType::GrayscaleAlpha);
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

    let (rgb_bytes, alpha) = extract_rgb_alpha_u8(raw, pixel_count, channels, has_alpha, is_grayscale);

    let rgb_bytes = apply_icc_transform_u8(rgb_bytes, icc_profile);

    let pixels: Vec<[f32; 3]> = rgb_bytes
        .chunks_exact(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();

    DecodedImage { pixels, width, height, alpha }
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

    let (rgb_u16, alpha) = extract_rgb_alpha_u16(raw, pixel_count, channels, has_alpha, is_grayscale);

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

    DecodedImage { pixels, width, height, alpha }
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

fn decode_jpeg(path: &Path, use_icc: bool) -> DecodedImage {
    let file = File::open(path).expect("Failed to open JPEG file");
    let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
    let raw = decoder.decode().expect("Failed to decode JPEG");
    let metadata = decoder.info().expect("Failed to get JPEG metadata");

    let width = metadata.width as usize;
    let height = metadata.height as usize;
    let pixel_count = width * height;

    let icc_profile = if use_icc {
        decoder.icc_profile()
    } else {
        None
    };

    let rgb_bytes = match metadata.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => raw,
        jpeg_decoder::PixelFormat::L8 => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for &g in &raw {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        jpeg_decoder::PixelFormat::CMYK32 => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for chunk in raw.chunks_exact(4) {
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
        jpeg_decoder::PixelFormat::L16 => {
            let mut rgb = Vec::with_capacity(pixel_count * 3);
            for chunk in raw.chunks_exact(2) {
                let g = (u16::from_ne_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
    };

    let rgb_bytes = apply_icc_transform_u8(rgb_bytes, &icc_profile);

    let pixels: Vec<[f32; 3]> = rgb_bytes
        .chunks_exact(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
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
