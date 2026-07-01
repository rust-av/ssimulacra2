mod icc;
#[cfg(feature = "video")]
mod video;

#[cfg(feature = "video")]
use self::video::*;
use clap::{Parser, Subcommand};
#[cfg(feature = "video")]
use ssimulacra2::MatrixCoefficients;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Compare two still images. Resolutions must be identical.
    Image {
        /// Source image
        #[arg(help = "Original unmodified image", value_hint = clap::ValueHint::FilePath)]
        source: PathBuf,

        /// Distorted image
        #[arg(help = "Distorted image", value_hint = clap::ValueHint::FilePath)]
        distorted: PathBuf,

        /// Disable ICC color profile conversion (assume sRGB)
        #[arg(long)]
        no_icc: bool,
    },
    /// Compare two videos. Resolutions and frame counts must be identical.
    #[cfg(feature = "video")]
    Video {
        /// Source video
        #[arg(help = "Original unmodified video", value_hint = clap::ValueHint::FilePath)]
        source: String,

        /// Distorted video
        #[arg(help = "Distorted video", value_hint = clap::ValueHint::FilePath)]
        distorted: String,

        /// How many worker threads to use for decoding & calculating scores.
        /// Note: Memory usage increases linearly with the number of workers.
        #[arg(long, short, verbatim_doc_comment)]
        frame_threads: Option<usize>,

        /// The amount of frames to skip.
        #[arg(long, default_value_t = 0)]
        skip_frames: usize,

        /// Limit the amount of frames to compare.
        #[arg(long)]
        frames: Option<usize>,

        /// How to increment current frame count; e.g. 10 will read every 10th frame.
        #[arg(long, short)]
        increment: Option<usize>,

        /// Whether to output a frame-by-frame graph of scores.
        #[arg(long, short)]
        graph: bool,

        /// Will output scores for every frame followed by the average at the end.
        #[arg(long, short)]
        verbose: bool,

        /// Source color matrix
        #[arg(long)]
        src_matrix: Option<String>,

        /// Source transfer characteristics
        #[arg(long)]
        src_transfer: Option<String>,

        /// Source color primaries
        #[arg(long)]
        src_primaries: Option<String>,

        /// The source is using full-range data
        #[arg(long)]
        src_full_range: bool,

        /// Distorted color matrix
        #[arg(long)]
        dst_matrix: Option<String>,

        /// Distorted transfer characteristics
        #[arg(long)]
        dst_transfer: Option<String>,

        /// Distorted color primaries
        #[arg(long)]
        dst_primaries: Option<String>,

        /// The distorted video is using full-range data
        #[arg(long)]
        dst_full_range: bool,
    },
}

fn main() {
    match Cli::parse().command {
        Commands::Image {
            source,
            distorted,
            no_icc,
        } => compare_images(&source, &distorted, !no_icc),
        #[cfg(feature = "video")]
        Commands::Video {
            source,
            distorted,
            frame_threads,
            skip_frames,
            frames,
            increment,
            graph,
            verbose,
            src_matrix,
            src_transfer,
            src_primaries,
            src_full_range,
            dst_matrix,
            dst_transfer,
            dst_primaries,
            dst_full_range,
        } => {
            let frame_threads = frame_threads.unwrap_or(1).max(1);
            let inc = increment.unwrap_or(1).max(1);
            let src_matrix = src_matrix
                .map(|i| parse_matrix(&i))
                .unwrap_or(MatrixCoefficients::Unspecified);
            let src_transfer = src_transfer
                .map(|i| parse_transfer(&i))
                .unwrap_or(TransferCharacteristic::Unspecified);
            let src_primaries = src_primaries
                .map(|i| parse_primaries(&i))
                .unwrap_or(ColorPrimaries::Unspecified);
            let dst_matrix = dst_matrix
                .map(|i| parse_matrix(&i))
                .unwrap_or(MatrixCoefficients::Unspecified);
            let dst_transfer = dst_transfer
                .map(|i| parse_transfer(&i))
                .unwrap_or(TransferCharacteristic::Unspecified);
            let dst_primaries = dst_primaries
                .map(|i| parse_primaries(&i))
                .unwrap_or(ColorPrimaries::Unspecified);
            compare_videos(
                &source,
                &distorted,
                frame_threads,
                skip_frames,
                frames,
                inc,
                graph,
                verbose,
                src_matrix,
                src_transfer,
                src_primaries,
                src_full_range,
                dst_matrix,
                dst_transfer,
                dst_primaries,
                dst_full_range,
            )
        }
    }
}

fn compare_images(source: &Path, distorted: &Path, use_icc: bool) {
    let src = icc::decode_image(source, use_icc);
    let dst = icc::decode_image(distorted, use_icc);

    let has_alpha = src.alpha.is_some() || dst.alpha.is_some();

    if has_alpha {
        let score = compute_alpha_aware_score(&src, &dst);
        println!("Score: {score:.8}");
    } else {
        let source_data = Rgb::new(
            src.pixels,
            src.width,
            src.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("Failed to process source into RGB");

        let distorted_data = Rgb::new(
            dst.pixels,
            dst.width,
            dst.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("Failed to process distorted into RGB");

        let result = compute_frame_ssimulacra2(source_data, distorted_data)
            .expect("Failed to calculate ssimulacra2");

        println!("Score: {result:.8}");
    }
}

fn compute_alpha_aware_score(src: &icc::DecodedImage, dst: &icc::DecodedImage) -> f64 {
    let src_alpha_default: Vec<f32>;
    let dst_alpha_default: Vec<f32>;

    let src_alpha = match &src.alpha {
        Some(a) => a.as_slice(),
        None => {
            src_alpha_default = vec![1.0; src.pixels.len()];
            &src_alpha_default
        }
    };
    let dst_alpha = match &dst.alpha {
        Some(a) => a.as_slice(),
        None => {
            dst_alpha_default = vec![1.0; dst.pixels.len()];
            &dst_alpha_default
        }
    };

    let mut worst = f64::MAX;

    for bg in [0.1_f32, 0.9_f32] {
        let src_composited = icc::composite_over_background(&src.pixels, src_alpha, bg);
        let dst_composited = icc::composite_over_background(&dst.pixels, dst_alpha, bg);

        let source_data = Rgb::new(
            src_composited,
            src.width,
            src.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("Failed to process source into RGB");

        let distorted_data = Rgb::new(
            dst_composited,
            dst.width,
            dst.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("Failed to process distorted into RGB");

        let score = compute_frame_ssimulacra2(source_data, distorted_data)
            .expect("Failed to calculate ssimulacra2");

        worst = worst.min(score);
    }

    worst
}
