use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../ssimulacra2/test_data")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}

fn score_with(args: &[&str]) -> f64 {
    let out = Command::new(env!("CARGO_BIN_EXE_ssimulacra2_rs"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "exit {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("Score: "))
        .expect("no Score line")
        .trim()
        .parse()
        .unwrap()
}

// tank_distorted.png is a 16-bit RGB PNG with an iCCP chunk: the 16-bit ICC
// path must work (regression: lcms2 panic "PixelFormat(...) has 6 bytes per
// pixel, but the input format has 2").
#[test]
fn icc_transform_on_16bit_png_does_not_crash() {
    let src = fixture("tank_source.png");
    let dst = fixture("tank_distorted.png");
    let with_icc = score_with(&["image", &src, &dst]);
    let without_icc = score_with(&["image", "--no-icc", &src, &dst]);
    // The embedded profile is sRGB: the conversion should be nearly neutral.
    assert!(
        (with_icc - without_icc).abs() < 0.25,
        "icc {with_icc} vs no-icc {without_icc}"
    );
}
