use super::*;

use crate::families::qwen3::from_yamf;
use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;

fn model() -> Qwen3 {
    // The mini fixture uses head_dim=16 for cheap tests; MLX's Metal
    // SDPA kernel supports only a small set of head_dims (32, 64, 96,
    // 128, 256) and crashes on 16. Force CPU here — MLX's CPU SDPA
    // fallback has no such constraint. Real Qwen3 uses head_dim=128
    // and runs fine on Metal; this only affects the tiny fixture.
    use nucleon_mlx::mlx_rs::Device;
    Device::set_default(&Device::cpu());
    from_yamf(&load_bytes(&mini().build()).unwrap()).unwrap()
}

#[test]
fn forward_produces_logits_of_the_right_shape() {
    // mini fixture: vocab 269, hidden 32. Feed 3 ids, expect logits
    // [seq, vocab] = [3, 269].
    let m = model();
    let ids = vec![0u32, 1u32, 2u32];
    let logits = m.forward(&ids).unwrap();
    assert_eq!(logits.shape(), &[3, 269]);
}

#[test]
fn single_token_forward_works() {
    // seq = 1 is the same code path as multi-token; the loop chapter
    // relies on it for the per-step generation call.
    let m = model();
    let logits = m.forward(&[5u32]).unwrap();
    assert_eq!(logits.shape(), &[1, 269]);
}

#[test]
fn empty_ids_refuses() {
    let err = model().forward(&[]).unwrap_err();
    assert!(err.to_string().contains("empty"));
}

#[test]
fn logits_are_finite() {
    // A generated garbage prompt from the mini fixture must still
    // produce all-finite logits; NaN/inf here would mean a norm or
    // matmul is exploding, which is what QK-norm and RMSNorm are
    // supposed to prevent.
    let m = model();
    let ids: Vec<u32> = (0..7).collect();
    let logits = m.forward(&ids).unwrap();
    for &v in logits.as_slice::<f32>() {
        assert!(v.is_finite(), "non-finite logit: {v}");
    }
}
