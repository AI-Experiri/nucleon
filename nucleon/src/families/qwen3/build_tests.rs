use super::*;

use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;

/// Same helper as forward_tests.rs — sets MLX default device to
/// CPU (Metal SDPA rejects head_dim=16 of the mini fixture),
/// restores GPU on drop. Duplicated here rather than shared so
/// each test file is self-contained.
struct CpuScope;
impl CpuScope {
    fn new() -> Self {
        use nucleon_mlx::mlx_rs::Device;
        Device::set_default(&Device::cpu());
        CpuScope
    }
}
impl Drop for CpuScope {
    fn drop(&mut self) {
        use nucleon_mlx::mlx_rs::Device;
        Device::set_default(&Device::gpu());
    }
}

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
fn untied_head_is_accepted_and_runs_forward() {
    // Round 3 L1 close: from_yamf accepts output.weight, AND the
    // untied branch of forward (Some(lm_head)) is actually
    // exercised — not just the assembly. Uses non-zero head values
    // so the untied path produces different logits than tied would.
    use crate::loader::formats::gguf::builder::{f32_bytes, GGML_F32};

    let _cpu = CpuScope::new();
    let untied: Vec<f32> = (0..32 * 269).map(|i| (i as f32) * 1e-4).collect();
    let bytes = mini()
        .tensor("output.weight", &[32, 269], GGML_F32, f32_bytes(&untied))
        .build();
    let y = load_bytes(&bytes).unwrap();
    let m = from_yamf(&y).unwrap();
    assert!(m.lm_head.is_some());

    // Run forward on the untied model — this exercises
    // forward.rs's `self.lm_head.as_ref().unwrap_or(&self.embed)`
    // Some branch, which was untested before.
    let logits = m.forward(&[1u32, 2u32]).unwrap();
    assert_eq!(logits.shape(), &[2, 269]);
    for &v in logits.as_slice::<f32>() {
        assert!(v.is_finite(), "non-finite logit in untied path: {v}");
    }
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
