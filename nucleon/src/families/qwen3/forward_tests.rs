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
fn out_of_range_id_refuses() {
    // MLX's gather does no bounds check — an out-of-range id
    // would silently produce garbage logits or read OOB memory.
    // Forward's own check catches it at the door.
    let _cpu = CpuScope::new();
    let m = model();
    let vocab = m.config().vocab_size;
    let err = m.forward(&[vocab]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("vocab_size"), "{msg}");
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

#[test]
fn rope_actually_rotates_along_seq_axis() {
    // Revert-proof for the RoPE axis bug caught in round 9.
    //
    // MLX's fast::rope treats axis 1 (after flattening leading dims)
    // as the sequence axis (fast.cpp:401,407). Correct wiring is
    // [1, H, seq, D] BEFORE rope; the previous (wrong) wiring was
    // [seq, H, D] which made MLX rotate the heads axis as if it were
    // positions.
    //
    // The mini fixture's constant attn_v/output weights collapse V
    // rows to identical values, which erases most RoPE-through-V
    // effects on the final logits (see comment below). BUT: with the
    // right axis order, differences in position-DEPENDENT rotations
    // survive at least as small numerical variations in logits when
    // rope_theta changes. With the WRONG axis (rotating heads), the
    // rotation is entirely position-independent, so changing theta
    // has literally no observable effect on any output value.
    //
    // So: pick two extreme theta values, run forward on a >1-token
    // prompt, and require at least ONE logit to differ. Under the
    // bug, all logits are byte-identical (rope_theta consumed only
    // for a heads-axis rotation whose effect cancels through the
    // uniform V). Under the fix, some differ (however small).
    let _cpu = CpuScope::new();

    let a = from_yamf(&load_bytes(&mini().build()).unwrap()).unwrap();

    let mut yamf_b = load_bytes(&mini().build()).unwrap();
    let crate::loader::FamilyConfig::Qwen3(cfg) = &mut yamf_b.family;
    cfg.rope_theta = 100.0;
    let b = from_yamf(&yamf_b).unwrap();

    let ids = vec![1u32, 5u32, 3u32];
    let la = a.forward(&ids).unwrap();
    let lb = b.forward(&ids).unwrap();
    let sa = la.as_slice::<f32>();
    let sb = lb.as_slice::<f32>();
    assert_eq!(sa.len(), sb.len());
    let any_diff = sa.iter().zip(sb.iter()).any(|(x, y)| (x - y).abs() > 0.0);
    assert!(
        any_diff,
        "logits byte-identical between rope_theta=1e6 and rope_theta=100 — RoPE is rotating the wrong axis (see round-9 bug)"
    );
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
