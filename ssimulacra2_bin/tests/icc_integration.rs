use std::path::PathBuf;
use std::process::Command;

fn fixture_path(name: &str) -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    path.push(name);
    path.to_string_lossy().into_owned()
}

fn run_score(args: &[&str]) -> f64 {
    let bin = env!("CARGO_BIN_EXE_ssimulacra2_rs");
    let output = Command::new(bin)
        .args(args)
        .output()
        .expect("Failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "Binary failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    stdout
        .trim()
        .strip_prefix("Score: ")
        .expect("Unexpected output format")
        .parse::<f64>()
        .expect("Failed to parse score")
}

#[test]
fn untagged_regression() {
    let score = run_score(&[
        "image",
        &fixture_path("untagged.png"),
        &fixture_path("untagged_distorted.png"),
    ]);
    assert!(
        (score - 95.30053).abs() < 0.001,
        "Untagged score {score} deviates from expected 95.30053"
    );
}

#[test]
fn srgb_tagged_matches_untagged() {
    let untagged = run_score(&[
        "image",
        &fixture_path("untagged.png"),
        &fixture_path("untagged_distorted.png"),
    ]);
    let tagged = run_score(&[
        "image",
        &fixture_path("srgb_tagged.png"),
        &fixture_path("srgb_tagged_distorted.png"),
    ]);
    assert!(
        (untagged - tagged).abs() < 1e-4,
        "sRGB tagged ({tagged}) should match untagged ({untagged})"
    );
}

#[test]
fn display_p3_differs_from_srgb() {
    let srgb = run_score(&[
        "image",
        &fixture_path("untagged.png"),
        &fixture_path("untagged_distorted.png"),
    ]);
    let p3 = run_score(&[
        "image",
        &fixture_path("display_p3.png"),
        &fixture_path("display_p3_distorted.png"),
    ]);
    assert!(
        (srgb - p3).abs() > 0.1,
        "Display P3 score ({p3}) should differ from sRGB ({srgb})"
    );
}

#[test]
fn adobe_rgb_differs_from_srgb() {
    let srgb = run_score(&[
        "image",
        &fixture_path("untagged.png"),
        &fixture_path("untagged_distorted.png"),
    ]);
    let adobe = run_score(&[
        "image",
        &fixture_path("adobe_rgb.png"),
        &fixture_path("adobe_rgb_distorted.png"),
    ]);
    assert!(
        (srgb - adobe).abs() > 0.01,
        "Adobe RGB score ({adobe}) should differ from sRGB ({srgb})"
    );
}

#[test]
fn no_icc_flag_matches_old_behavior() {
    let untagged = run_score(&[
        "image",
        &fixture_path("untagged.png"),
        &fixture_path("untagged_distorted.png"),
    ]);
    let no_icc = run_score(&[
        "image",
        "--no-icc",
        &fixture_path("display_p3.png"),
        &fixture_path("display_p3_distorted.png"),
    ]);
    assert!(
        (untagged - no_icc).abs() < 1e-4,
        "--no-icc on Display P3 ({no_icc}) should match untagged sRGB ({untagged})"
    );
}

#[test]
fn display_p3_reference_score() {
    let score = run_score(&[
        "image",
        &fixture_path("display_p3.png"),
        &fixture_path("display_p3_distorted.png"),
    ]);
    assert!(
        (score - 94.31656).abs() < 0.001,
        "Display P3 score {score} deviates from reference 94.31656"
    );
}

#[test]
fn adobe_rgb_reference_score() {
    let score = run_score(&[
        "image",
        &fixture_path("adobe_rgb.png"),
        &fixture_path("adobe_rgb_distorted.png"),
    ]);
    assert!(
        (score - 94.89051).abs() < 0.001,
        "Adobe RGB score {score} deviates from reference 94.89051"
    );
}
