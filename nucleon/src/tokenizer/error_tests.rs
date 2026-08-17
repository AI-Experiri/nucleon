use super::*;

#[test]
fn display_carries_the_reason() {
    let e = TokenizerError::Build {
        reason: "merge references unknown token".to_string(),
    };
    let s = e.to_string();
    assert!(s.contains("build failed"));
    assert!(s.contains("merge references unknown token"));
}

#[test]
fn constructors_wrap_any_display_type() {
    // the call sites pass HF-crate errors; any Display works
    assert!(matches!(
        TokenizerError::build("boom"),
        TokenizerError::Build { .. }
    ));
    assert!(matches!(
        TokenizerError::encode("boom"),
        TokenizerError::Encode { .. }
    ));
    assert!(matches!(
        TokenizerError::decode("boom"),
        TokenizerError::Decode { .. }
    ));
}

#[test]
fn each_variant_names_its_operation() {
    let enc = TokenizerError::Encode {
        reason: "x".to_string(),
    };
    let dec = TokenizerError::Decode {
        reason: "x".to_string(),
    };
    assert!(enc.to_string().starts_with("encode failed"));
    assert!(dec.to_string().starts_with("decode failed"));
}
