use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::clamp;
use rand::Rng;
use ssimulacra2::{
    compute_frame_ssimulacra2, xyb_to_planar, Blur, ColorPrimaries, Frame, MatrixCoefficients,
    Plane, TransferCharacteristic, Yuv, YuvConfig,
};
use yuvxyb::{LinearRgb, Xyb};

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
    let mut rng = rand::rng();
    for (i, plane) in data.planes.iter_mut().enumerate() {
        for val in plane.data_origin_mut().iter_mut() {
            *val = rng.random_range(if full_range {
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
    let mut rng = rand::rng();
    let mut planes = [
        input.data()[0].clone(),
        input.data()[1].clone(),
        input.data()[2].clone(),
    ];
    for plane in &mut planes {
        for pix in plane.data_origin_mut() {
            *pix = clamp(i16::from(*pix) + rng.random_range(-16..=16), 0, 255) as u8;
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
        let mut out = std::array::from_fn(|_| vec![0.0f32; width * height]);

        b.iter(|| blur.blur(black_box(&image), black_box(&mut out)));
    });
}

fn create_linearrgb_from_image(path: &str) -> LinearRgb {
    let img = image::open(path).unwrap();
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();

    let mut data = vec![[0.0f32; 3]; (width * height) as usize];
    for (i, pixel) in img.pixels().enumerate() {
        data[i] = [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        ];
    }

    LinearRgb::new(data, width as usize, height as usize).expect("이미지 변환 실패")
}

// downscale_by_2
fn bench_downscale_by_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("downscale");
    group.measurement_time(std::time::Duration::from_secs(9));

    group.bench_function("downscale_by_2", |b| {
        // load the image
        let image = create_linearrgb_from_image("test_data/tank_source.png");

        b.iter(|| ssimulacra2::downscale_by_2(black_box(&image)))
    });
    group.finish();
}

fn bench_image_multiply(c: &mut Criterion) {
    c.bench_function("image_multiply", |b| {
        let image = create_linearrgb_from_image("test_data/tank_source.png");
        let width = image.width();
        let height = image.height();

        let img1 = xyb_to_planar(&Xyb::from(image.clone()));
        let img2 = img1.clone();

        let mut out = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];

        b.iter(|| {
            ssimulacra2::image_multiply(black_box(&img1), black_box(&img2), black_box(&mut out))
        })
    });
}

fn bench_ssim_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim_map");
    group.measurement_time(std::time::Duration::from_secs(9));

    group.bench_function("standard", |b| {
        let source_image = create_linearrgb_from_image("test_data/tank_source.png");
        let distorted_image = create_linearrgb_from_image("test_data/tank_distorted.png");
        let width = source_image.width();
        let height = source_image.height();

        let mut img1_xyb = Xyb::from(source_image);
        let mut img2_xyb = Xyb::from(distorted_image);

        for pix in img1_xyb.data_mut().iter_mut() {
            pix[2] = (pix[2] - pix[1]) + 0.55;
            pix[0] = (pix[0]).mul_add(14.0, 0.42);
            pix[1] += 0.01;
        }
        for pix in img2_xyb.data_mut().iter_mut() {
            pix[2] = (pix[2] - pix[1]) + 0.55;
            pix[0] = (pix[0]).mul_add(14.0, 0.42);
            pix[1] += 0.01;
        }

        let img1_planar = xyb_to_planar(&img1_xyb);
        let img2_planar = xyb_to_planar(&img2_xyb);

        let mut blur = Blur::new(width, height);

        let mut mul = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut sigma1_sq = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut sigma2_sq = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut sigma12 = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut mu1 = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut mu2 = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];

        ssimulacra2::image_multiply(&img1_planar, &img1_planar, &mut mul);
        blur.blur(&mul, &mut sigma1_sq).expect("blur failed");

        ssimulacra2::image_multiply(&img2_planar, &img2_planar, &mut mul);
        blur.blur(&mul, &mut sigma2_sq).expect("blur failed");

        ssimulacra2::image_multiply(&img1_planar, &img2_planar, &mut mul);
        blur.blur(&mul, &mut sigma12).expect("blur failed");

        blur.blur(&img1_planar, &mut mu1).expect("blur failed");
        blur.blur(&img2_planar, &mut mu2).expect("blur failed");

        b.iter(|| {
            ssimulacra2::ssim_map(
                black_box(width),
                black_box(height),
                black_box(&mu1),
                black_box(&mu2),
                black_box(&sigma1_sq),
                black_box(&sigma2_sq),
                black_box(&sigma12),
            )
        });
    });

    group.finish();
}

fn bench_xyb_to_planar(c: &mut Criterion) {
    c.bench_function("xyb_to_planar", |b| {
        let image = create_linearrgb_from_image("test_data/tank_source.png");
        let xyb = Xyb::from(image.clone());

        b.iter(|| xyb_to_planar(black_box(&xyb)))
    });
}

// criterion_group!(benches, bench_ssimulacra2, bench_blur);
criterion_group!(
    benches,
    bench_ssimulacra2,
    bench_blur,
    bench_downscale_by_2,
    bench_image_multiply,
    bench_ssim_map,
    bench_xyb_to_planar
);
criterion_main!(benches);
