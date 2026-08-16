use super::*;
use crate::loader::formats::gguf::builder::{f32_bytes, quantize_q8_0_ref};

#[test]
fn f32_passes_through_bit_exactly() {
    let vals = [0.0f32, 1.5, -2.25, f32::MIN_POSITIVE];
    let out = dequantize(GGML_F32, &f32_bytes(&vals), 4, 4, "t").unwrap();
    for (a, b) in vals.iter().zip(&out) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn q8_0_dequant_matches_the_formula_exactly() {
    // hand-built block: d = 2.0 (exact in f16), q = 0, 1, -2, ...
    let mut block = Vec::new();
    block.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    let qs: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
    for q in &qs {
        block.push(*q as u8);
    }
    let out = dequantize(GGML_Q8_0, &block, 32, 32, "t").unwrap();
    for (q, y) in qs.iter().zip(&out) {
        assert_eq!(*y, 2.0 * f32::from(*q));
    }
}

#[test]
fn q8_0_reference_roundtrip_recovers_integers_exactly() {
    // every block contains -127, so amax = 127 and d = 1.0 exactly:
    // integer values survive the roundtrip bit-for-bit
    let vals: Vec<f32> = (0..64).map(|i| ((i % 32) * 4) as f32 - 127.0).collect();
    let bytes = quantize_q8_0_ref(&vals);
    let out = dequantize(GGML_Q8_0, &bytes, 64, 32, "t").unwrap();
    assert_eq!(vals, out);
}

#[test]
fn q8_0_rejects_non_multiple_of_32() {
    match tensor_byte_len(GGML_Q8_0, 66, 33, "t") {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("32"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn wrong_byte_count_is_refused() {
    let bytes = f32_bytes(&[1.0, 2.0]);
    match dequantize(GGML_F32, &bytes, 3, 3, "t") {
        Err(LoaderError::Structure { reason }) => assert!(reason.contains("bytes"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unsupported_type_names_itself_and_the_supported_list() {
    match tensor_byte_len(2, 32, 32, "blk.0.ffn_up.weight") {
        Err(e @ LoaderError::UnsupportedTensorType { type_id: 2, .. }) => {
            let msg = e.to_string();
            assert!(
                msg.contains("blk.0.ffn_up.weight") && msg.contains("Q8_0"),
                "{msg}"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn inconsistent_elems_and_rows_are_refused() {
    match tensor_byte_len(GGML_Q8_0, 33, 32, "t") {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("rows"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn f32_nan_or_inf_values_are_refused() {
    let vals = [1.0f32, f32::NAN];
    match dequantize(GGML_F32, &f32_bytes(&vals), 2, 2, "t") {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("non-finite"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
    let vals = [f32::INFINITY, 0.0];
    match dequantize(GGML_F32, &f32_bytes(&vals), 2, 2, "t") {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("non-finite"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn q8_0_nan_or_inf_scale_is_refused() {
    // one block: NaN scale in f16 (0x7E00 is a quiet NaN) + 32 zeros
    let mut block = Vec::new();
    block.extend_from_slice(&half::f16::from_bits(0x7E00).to_le_bytes());
    block.extend(std::iter::repeat_n(0u8, 32));
    match dequantize(GGML_Q8_0, &block, 32, 32, "t") {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("non-finite"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}
