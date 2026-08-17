use super::*;

#[test]
fn version_triple_names_all_three() {
    let s = version_triple();
    assert!(s.contains("mlx-rs"));
    assert!(s.contains("mlx-c"));
    assert!(s.contains("MLX"));
}

#[test]
fn version_pins_are_the_ones_the_book_documents() {
    assert_eq!(mlx_rs_version(), "0.25.3");
    assert_eq!(mlx_c_version(), "0.5.0");
    assert_eq!(mlx_version(), "0.30.6");
}
