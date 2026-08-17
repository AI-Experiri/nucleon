use super::*;

use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;

fn yamf() -> crate::loader::Yamf {
    load_bytes(&mini().build()).unwrap()
}

#[test]
fn builds_from_the_mini_fixture() {
    let y = yamf();
    let m = from_yamf(&y).unwrap();
    assert_eq!(m.config().num_hidden_layers, 1);
    assert_eq!(m.config().hidden_size, 32);
    assert_eq!(m.config().num_attention_heads, 2);
    assert_eq!(m.config().num_key_value_heads, 1);
    assert_eq!(m.config().head_dim, 16);
    // 0.6B-style: tied lm head, no output.weight; mini() follows suit.
    assert!(m.lm_head.is_none());
    assert_eq!(m.blocks.len(), 1);
}

#[test]
fn untied_head_is_accepted() {
    use crate::loader::formats::gguf::builder::{f32_bytes, GGML_F32};

    let untied: Vec<f32> = vec![0.0; 32 * 269];
    let bytes = mini()
        .tensor("output.weight", &[32, 269], GGML_F32, f32_bytes(&untied))
        .build();
    let y = load_bytes(&bytes).unwrap();
    let m = from_yamf(&y).unwrap();
    assert!(m.lm_head.is_some());
}

#[test]
fn wrong_shape_tensor_refuses() {
    // Yamf's fields are public; a hand-built bundle could substitute
    // a wrong-shaped tensor for a real weight. from_yamf must refuse
    // at its own border rather than let it crash later in forward.
    let mut y = yamf();
    let bad = nucleon_mlx::array_from_f32(vec![0.0; 4], &[2, 2]);
    y.tensors.insert("token_embd.weight".to_string(), bad);
    let Err(err) = from_yamf(&y) else {
        panic!("wrong-shaped token_embd.weight must refuse")
    };
    let msg = err.to_string();
    assert!(msg.contains("token_embd"), "{msg}");
    assert!(msg.contains("expected"), "{msg}");
}

#[test]
fn missing_required_tensor_refuses() {
    let mut y = yamf();
    // pluck a tensor from the map to simulate a broken bundle
    y.tensors.remove("token_embd.weight").unwrap();
    let Err(err) = from_yamf(&y) else {
        panic!("missing token_embd must refuse");
    };
    assert!(err.to_string().contains("token_embd"));
    assert!(err.to_string().contains("absent"));
}
