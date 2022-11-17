use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::clamp;
use rand::Rng;
use ssimulacra2::{
    compute_frame_ssimulacra2, Blur, ColorPrimaries, Frame, MatrixCoefficients, Plane,
    TransferCharacteristic, Yuv, YuvConfig,
};

fn make_yuv(
    ss: (u8, u8),
    full_range: bool,
    mc: MatrixCoefficients,
    tc: TransferCharacteristic,
    cp: ColorPrimaries,
) -> Yuv<u8> {
    let y_dims = (320usize, 240usize);
    let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
    let mut data: Frame<u8> = Frame {
        planes: [
            Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
        ],
    };
    let mut rng = rand::thread_rng();
    for (i, plane) in data.planes.iter_mut().enumerate() {
        for val in plane.data_origin_mut().iter_mut() {
            *val = rng.gen_range(if full_range {
                0..=255
            } else if i == 0 {
                16..=235
            } else {
                16..=240
            });
        }
    }
    Yuv::new(
        data,
        YuvConfig {
            bit_depth: 8,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
            color_primaries: cp,
        },
    )
    .unwrap()
}

fn distort_yuv(input: &Yuv<u8>) -> Yuv<u8> {
    let mut rng = rand::thread_rng();
    let mut planes = [
        input.data()[0].clone(),
        input.data()[1].clone(),
        input.data()[2].clone(),
    ];
    for plane in &mut planes {
        for pix in plane.data_origin_mut() {
            *pix = clamp(i16::from(*pix) + rng.gen_range(-16..=16), 0, 255) as u8;
        }
    }
    let data: Frame<u8> = Frame { planes };
    Yuv::new(data, input.config()).unwrap()
}

fn bench_ssimulacra2(c: &mut Criterion) {
    c.bench_function("ssimulacra2", |b| {
        let input = make_yuv(
            (0, 0),
            true,
            MatrixCoefficients::BT709,
            TransferCharacteristic::BT1886,
            ColorPrimaries::BT709,
        );
        let distorted = distort_yuv(&input);
        b.iter(|| compute_frame_ssimulacra2(black_box(&input), black_box(&distorted)).unwrap())
    });
}

fn read_image(path: &str) -> ([Vec<f32>; 3], usize, usize) {
    // Read in test_data/tank_source.png
    let img = image::open(path).unwrap();

    let img = match img {
        image::DynamicImage::ImageRgb8(img) => img,
        x => x.to_rgb8(),
    };

    let (width, height) = img.dimensions();

    // Convert ImageBuffer to [Vec<f32>; 3]
    let mut img_vec = [Vec::new(), Vec::new(), Vec::new()];
    for (_i, pixel) in img.pixels().enumerate() {
        img_vec[0].push(pixel[0] as f32);
        img_vec[1].push(pixel[1] as f32);
        img_vec[2].push(pixel[2] as f32);
    }

    (img_vec, width as usize, height as usize)
}

fn bench_blur(c: &mut Criterion) {
    c.bench_function("blur", |b| {
        let (image, width, height) = read_image("test_data/tank_source.png");

        // Blur the image
        let mut blur = Blur::new(width, height);

        b.iter(|| blur.blur(black_box(&image)))
    });
}

criterion_group!(benches, bench_ssimulacra2, bench_blur);
criterion_main!(benches);
