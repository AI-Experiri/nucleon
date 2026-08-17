# 0011 — 2026-08-17 — the qwen3 family block

## What happened

User asked to implement the Qwen3-0.6B forward pass in code
following the review-convergence protocol they specified: `cc-qwen`
reviews (a fish function wrapping claude-code CLI against a
DeepSeek Qwen-family endpoint), Sonnet subagent validates each
finding, Opus (me) applies + revert-proof-verifies. Convergence
rule per CLAUDE.md: 3 consecutive stable rounds (no real bugs).

## The code, in one place

Six files, ~350 lines including tests:

- `nucleon/src/families/mod.rs` — barrel
- `nucleon/src/families/qwen3/mod.rs` — barrel exposing
  `from_yamf`, `Qwen3`, `FamilyError` only
- `nucleon/src/families/qwen3/error.rs` — `FamilyError` with
  Build/Forward variants
- `nucleon/src/families/qwen3/build.rs` — `Qwen3` struct + `Block`
  substruct; `from_yamf` clones the loader's Yamf tensor handles
  (refcounted `Array`s) into named fields, validating each tensor's
  shape against config-derived dimensions
- `nucleon/src/families/qwen3/forward.rs` — the pass in MLX ops:
  RMSNorm → Q/K/V matmuls with pre-transposed weights → head split
  → QK-norm (RMSNorm on last axis, shared weight across heads) →
  RoPE (traditional=false, base 1e6, offset=0) → SDPA (causal,
  scale 1/√head_dim, MLX handles GQA broadcast) → concat + output
  projection → residual, then SwiGLU MLP + residual. Repeated per
  block. Final RMSNorm + lm head (tied or untied via
  `unwrap_or(&self.embed)`).
- Sibling `_tests.rs` files for build/forward/error.

Added one re-export to `nucleon-mlx`: `nn::silu` (SwiGLU), and
`pub use mlx_rs` so callers can reach `ScaledDotProductAttentionMask`
without a second crate dep.

## Notes on the compute layer

- MLX's Metal SDPA rejects head_dim < 32; mini fixture uses
  head_dim=16 for cheap tests, so all forward tests force CPU
  device via a `CpuScope` RAII guard. Real Qwen3-0.6B runs
  head_dim=128 on Metal with no such constraint. `quality.sh`
  passes `--test-threads=1` because `set_default_device` is
  process-global.

- Q/K/V projections use `x @ W^T` since GGUF stores weights in
  `[out, in]` row-major. Per-block `.transpose()` calls rebuild
  lazy metadata nodes each forward call, no data copy; accepted
  as non-optimization per review r3.

- SDPA takes rank-4 `[batch, n_heads, seq, head_dim]`. We add a
  batch dim of 1 (single-batch inference), transpose seq and
  heads via `transpose_axes([0, 2, 1, 3])`, inverse afterwards,
  then reshape to `[seq, q_rows]` for the output projection.

## Review rounds

Six rounds, all reviewers independent Opus/Sonnet subagents or
cc-qwen. Every code fix revert-proof-verified (revert change,
watch named test fail, restore).

- **r1** (cc-qwen: CLEAN on math; 4 doc/coverage items; all
  Sonnet-CONFIRMED): from_yamf claimed shape validation it didn't
  perform → added per-tensor shape check; value-correctness of
  RoPE/QK-norm/scale/GQA/tied-head not asserted anywhere → tried
  a RoPE-varying-theta test, failed because mini fixture's
  uniform weights make V position-invariant (softmax(any)·V=V);
  documented the gap explicitly and deferred to loop chapter's
  HF-oracle golden test; Metal head_dim=128 unverified comment
  softened; test leaked global device state → CpuScope RAII.
- **r2** (cc-qwen: CLEAN on math + shape/RoPE/SDPA/transpose
  verified against real MLX C++ source; 1 Low; Sonnet-CONFIRMED):
  MLX's gather has no bounds check → forward now refuses ids ≥
  vocab_size at its own door.
- **r3** (cc-qwen: CLEAN on 10 hunt-list angles including memory
  retention, lazy graph traps, f32 overflow, dtype consistency,
  transpose-in-loop being lazy; 2 Lows on coverage; Sonnet found
  1 real thing on top): nucleon-mlx used inline
  `#[cfg(test)] mod tests { ... }` — CLAUDE.md HARD rule
  violation, moved to sibling `lib_tests.rs`. Untied lm_head
  branch was accepted but never exercised in forward — extended
  test to run forward on the untied model.
- **r4** (cc-qwen: 1 Low + 1 Nit; Sonnet WEAKENED one, CONFIRMED
  the Nit): Dtype re-export in nucleon-mlx was dead, removed;
  row_major_offset pub is acceptable cross-crate glue, not a
  rule violation. **Stable round 1**.
- **r5** (cc-qwen: 1 Low, theoretical): unchecked u32→i32 cast on
  vocab_size and seq — WEAKENED to unreachable in practice
  (needs 8TB Array). Accepted per Sonnet's recommendation.
  **Stable round 2**.
- **r6** (cc-qwen: 4 Lows, all doc drift or diagnostic; all
  Sonnet-CONFIRMED): "op wrappers move here" doc stale; "moves
  the Yamf tensors" said the wrong thing (clones handles);
  "16/8 = 2" comment stated a ratio the fixture doesn't run;
  Forward errors lost block index → wrapped in per-iteration
  closure so errors report "layer N: <mlx string>". No
  correctness defect. **Stable round 3.**

## CONVERGED

Rounds 4, 5, 6 all found no real bugs — doc/comment/diagnostic
items only. Per CLAUDE.md's convergence rule (3 consecutive
stable rounds, only already-assessed items or non-bugs), the
block is converged.

## Final state

- 13 tests passing (11 family + 2 nucleon-mlx)
- Quality gate green
- All 6 rounds of code fixes revert-proof-verified
- Known coverage gap: value correctness of the forward pass
  (RoPE actually rotating with correct theta, QK-norm ordering,
  attn scale, GQA broadcast, tied-head matmul) is NOT asserted
  by this module's unit tests. The book explicitly acknowledges
  this ("no fixture can tell correct attention from subtly-
  wrong attention") and defers to the loop chapter's HF-oracle
  golden test.

## Follow-ups (out of this block)

- The Loop (chapter 10): wire the family, tokenizer, and sampler
  into a generation loop; add the HF-transformers golden test
  that proves value correctness on the real Qwen3-0.6B GGUF.
- The Sampler (chapter 11): greedy / temperature / top-k / top-p.
- The CLI (chapter 12): ChatML template rendering.
- (Part III) The Cache: KV storage per past position × KV heads;
  first measured improvement after the naive loop.
- Optional micro-opt (r3 L2): hoist per-block weight transposes
  from forward() to from_yamf(); non-defect.
