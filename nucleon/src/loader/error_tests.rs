use super::*;

#[test]
fn refusals_state_found_and_supported() {
    let msg = LoaderError::UnsupportedVersion { found: 4 }.to_string();
    assert!(msg.contains('4') && msg.contains("supports 3"), "{msg}");

    let msg = LoaderError::UnsupportedArchitecture {
        found: "llama".into(),
    }
    .to_string();
    assert!(msg.contains("llama") && msg.contains("qwen3"), "{msg}");

    let msg = LoaderError::UnsupportedTensorType {
        name: "blk.0.ffn_up.weight".into(),
        type_id: 12,
    }
    .to_string();
    assert!(msg.contains("12") && msg.contains("Q8_0"), "{msg}");
}

#[test]
fn big_endian_version_gets_named() {
    let msg = LoaderError::UnsupportedVersion { found: 50_331_648 }.to_string();
    assert!(msg.contains("big-endian"), "{msg}");
}

#[test]
fn wrong_shape_shows_both_shapes() {
    let msg = LoaderError::WrongShape {
        name: "token_embd.weight".into(),
        want: vec![10, 8],
        found: vec![8, 10],
    }
    .to_string();
    assert!(msg.contains("[10, 8]") && msg.contains("[8, 10]"), "{msg}");
}
