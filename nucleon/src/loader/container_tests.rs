use super::*;
use crate::loader::gguf_builder::{f32_bytes, GgufBuilder, GGML_F32};

fn tiny() -> Vec<u8> {
    GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_bool("tokenizer.ggml.add_bos_token", false)
        .kv_arr_str("tokenizer.ggml.tokens", &["a", "b"])
        .kv_arr_i32("tokenizer.ggml.token_type", &[1, 3])
        .tensor("t.weight", &[4, 2], GGML_F32, f32_bytes(&[0.0; 8]))
        .build()
}

#[test]
fn parses_header_metadata_and_tensor_infos() {
    let c = parse(&tiny()).unwrap();
    assert_eq!(c.alignment, 32);
    assert_eq!(c.tensors.len(), 1);
    assert_eq!(c.tensors[0].name, "t.weight");
    assert_eq!(c.tensors[0].dims, vec![4, 2]); // ne order, unreversed here
    assert_eq!(c.tensors[0].offset, 0);
    assert_eq!(
        c.metadata.get("general.architecture"),
        Some(&MetaValue::Str("qwen3".into()))
    );
    assert_eq!(
        c.metadata.get("qwen3.block_count"),
        Some(&MetaValue::U32(2))
    );
    assert_eq!(
        c.metadata.get("tokenizer.ggml.add_bos_token"),
        Some(&MetaValue::Bool(false))
    );
    match c.metadata.get("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(items)) => assert_eq!(items.len(), 2),
        other => panic!("tokens: {other:?}"),
    }
    // data region starts aligned and inside the file
    assert_eq!(c.data_start % 32, 0);
    assert!(c.data_start + 32 <= c.file_len);
}

#[test]
fn f32_metadata_survives_bit_exactly() {
    let bytes = GgufBuilder::new()
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .build();
    let c = parse(&bytes).unwrap();
    match c.metadata.get("qwen3.attention.layer_norm_rms_epsilon") {
        Some(MetaValue::F32(v)) => assert_eq!(v.to_bits(), 1e-6f32.to_bits()),
        other => panic!("eps: {other:?}"),
    }
}

#[test]
fn refuses_wrong_magic() {
    let mut b = tiny();
    b[0] = b'X';
    match parse(&b) {
        Err(LoaderError::NotGguf { .. }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_version_2_and_big_endian_3() {
    let b = GgufBuilder::new().version(2).build();
    match parse(&b) {
        Err(LoaderError::UnsupportedVersion { found: 2 }) => {}
        other => panic!("{other:?}"),
    }
    let b = GgufBuilder::new().version(50_331_648).build();
    match parse(&b) {
        Err(e @ LoaderError::UnsupportedVersion { .. }) => {
            assert!(e.to_string().contains("big-endian"))
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_truncated_file() {
    let full = tiny();
    // cut inside the metadata section
    match parse(&full[..40]) {
        Err(LoaderError::Truncated { .. }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_lying_string_length() {
    // header + one kv whose key claims to be enormous
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensors
    b.extend_from_slice(&1u64.to_le_bytes()); // kvs
    b.extend_from_slice(&u64::MAX.to_le_bytes()); // key length: a lie
    match parse(&b) {
        Err(LoaderError::Truncated { .. }) | Err(LoaderError::Structure { .. }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_bad_bool_and_unknown_meta_type() {
    // hand-build: one kv "k" of type bool with byte 7
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.push(b'k');
    b.extend_from_slice(&7u32.to_le_bytes()); // bool type
    b.push(7); // invalid bool byte
    match parse(&b) {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("bool"), "{reason}"),
        other => panic!("{other:?}"),
    }

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.push(b'k');
    b.extend_from_slice(&13u32.to_le_bytes()); // beyond spec's 0..=12
    match parse(&b) {
        Err(LoaderError::UnknownMetaType { type_id: 13, .. }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_structural_tensor_lies() {
    // duplicate tensor names
    let b = GgufBuilder::new()
        .tensor("t", &[32], GGML_F32, f32_bytes(&[0.0; 32]))
        .tensor("t", &[32], GGML_F32, f32_bytes(&[0.0; 32]))
        .build();
    match parse(&b) {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("duplicate"), "{reason}"),
        other => panic!("{other:?}"),
    }

    // a 65-byte tensor name
    let long = "n".repeat(65);
    let b = GgufBuilder::new()
        .tensor(&long, &[32], GGML_F32, f32_bytes(&[0.0; 32]))
        .build();
    match parse(&b) {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("64"), "{reason}"),
        other => panic!("{other:?}"),
    }

    // a zero dimension
    let b = GgufBuilder::new()
        .tensor("t", &[0, 4], GGML_F32, Vec::new())
        .build();
    match parse(&b) {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("zero"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_duplicate_metadata_keys() {
    let b = GgufBuilder::new().kv_u32("k", 1).kv_u32("k", 2).build();
    match parse(&b) {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("duplicate"), "{reason}"),
        other => panic!("{other:?}"),
    }
}
