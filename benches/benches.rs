use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::clamp;
use rand::Rng;
use ssimulacra2::{
    compute_frame_ssimulacra2, ColorPrimaries, Frame, MatrixCoefficients, Plane,
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

criterion_group!(benches, bench_ssimulacra2);
criterion_main!(benches);
