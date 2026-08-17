use super::*;

use crate::families::qwen3::from_yamf;
use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;

/// Scope guard: sets MLX's default device to CPU on construction
/// and restores GPU on drop. Tests that mutate the global default
/// use this so any test running afterwards sees a clean default
/// again (parallel-safe only under --test-threads=1, which
/// quality.sh sets — the mutation is process-global, not thread-
/// local, and MLX has no scoped-device API in this crate version).
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

/// The mini fixture uses head_dim=16 for cheap tests; MLX's Metal
/// SDPA kernel supports only head_dims in {32, 64, 80, 96, 128,
/// 256} and SIGSEGVs on 16. CPU SDPA has no such constraint. The
/// production model runs head_dim=128 on Metal; that path is
/// proven not here but by the loop chapter's end-to-end golden
/// test against HF transformers on the real Qwen3-0.6B GGUF.
fn model() -> Qwen3 {
    from_yamf(&load_bytes(&mini().build()).unwrap()).unwrap()
}

#[test]
fn forward_produces_logits_of_the_right_shape() {
    // mini fixture: vocab 269, hidden 32. Feed 3 ids, expect logits
    // [seq, vocab] = [3, 269].
    let _cpu = CpuScope::new();
    let m = model();
    let ids = vec![0u32, 1u32, 2u32];
    let logits = m.forward(&ids).unwrap();
    assert_eq!(logits.shape(), &[3, 269]);
}

#[test]
fn single_token_forward_works() {
    // seq = 1 is the same code path as multi-token; the loop chapter
    // relies on it for the per-step generation call.
    let _cpu = CpuScope::new();
    let m = model();
    let logits = m.forward(&[5u32]).unwrap();
    assert_eq!(logits.shape(), &[1, 269]);
}

#[test]
fn empty_ids_refuses() {
    let _cpu = CpuScope::new();
    let err = model().forward(&[]).unwrap_err();
    assert!(err.to_string().contains("empty"));
}

#[test]
fn logits_are_finite() {
    // A generated garbage prompt from the mini fixture must still
    // produce all-finite logits; NaN/inf here would mean a norm or
    // matmul is exploding, which is what QK-norm and RMSNorm are
    // supposed to prevent.
    let _cpu = CpuScope::new();
    let m = model();
    let ids: Vec<u32> = (0..7).collect();
    let logits = m.forward(&ids).unwrap();
    for &v in logits.as_slice::<f32>() {
        assert!(v.is_finite(), "non-finite logit: {v}");
    }
}

// COVERAGE GAP acknowledged (finding B, review round 1):
//
// The value-correctness of the wiring — RoPE traditional=false,
// attn_scale=1/sqrt(head_dim), qk-norm-after-head-split-before-
// RoPE, GQA 2:1 broadcast, tied-head matmul — is NOT unit-tested
// here. Structural tests (shape, finiteness, refusal) cannot
// detect a subtly-wrong forward pass. Book 9.9 states this
// explicitly: "no fixture can tell correct attention from subtly-
// wrong attention. It is the golden test in the loop chapter."
//
// A wiring-value test needs an oracle. The two options considered
// and their costs:
//   1) Compare against HF transformers on the real Qwen3-0.6B
//      GGUF (the loop chapter's plan). Requires the real model
//      loaded and a Python oracle. Belongs in tests/ as an
//      end-to-end integration test with the loop.
//   2) A value-varying tiny fixture. The current mini() fixture
//      has all-uniform attn_q/k/v weights, which make V rows
//      position-invariant after rmsnorm+matmul (embed values
//      linear in row index ⇒ rmsnorm+sum ≈ constant). This
//      erases RoPE's effect on the final logits — softmax(any) ·
//      identical-V = identical-V. A meaningful RoPE test would
//      need a redesigned fixture that ripples through every
//      loader test.
//
// Deferring to option 1 in the loop chapter. This file pins the
// public contract (shapes, refusals, all-finite) that the golden
// test will build on.
