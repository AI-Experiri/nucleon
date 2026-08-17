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

## Initial "converged" call was wrong

I declared convergence after r4/r5/r6 on a lax reading (doc-only
and cleanup items counted as "stable"). User pushed back:
"converged?" Under a strict reading of CLAUDE.md — "clean OR only
already-assessed items" — r4 and r6 both had new items applied
and don't qualify. Correct reset. Continued to r7+.

## Rounds 7-12 (strict-convergence pass)

- **r7** — cc-qwen: CLEAN (functional); 1 Nit (Unicode arrow in
  `version_triple()` display string). Sonnet REJECTED (writing-style
  rule scoped to prose only, not code display strings).
  **Strict stable 1/3.**
- **r8** — cc-qwen: CLEAN. Sonnet sanity-check on the CLEAN
  verdict came back CONFIRMED CLEAN (noted only a cross-file
  assumption already enforced at loader gate).
  **Strict stable 2/3.**
- **r9** — cc-qwen: CLEAN. Sonnet's cold-read sanity check found a
  **HIGH bug that survived 8 previous rounds**: MLX's `fast::rope`
  flattens leading dims and treats axis 1 as sequence
  (`fast.cpp:401,407`). Old code passed RoPE input as
  `[seq, n_heads, head_dim]` — MLX rotated the n_heads axis as if
  it were positions. Every seq position got the same rotation;
  every head got a meaningless "position" rotation. Silent wrong
  logits through the whole forward pass. Fixed by transposing
  Q/K/V to `[1, H, seq, D]` BEFORE QK-norm and RoPE (matches
  HF `modeling_qwen3.py` order). Revert-proof test
  `rope_actually_rotates_along_seq_axis` compares logits at two
  rope_theta values — under the bug they are byte-identical (rope
  effect is position-independent so consumes zero downstream);
  under the fix they differ. **Strict stable RESET to 0.**
- **r10** — cc-qwen: CLEAN. Verified the r9 fix against HF's
  reshape→transpose(1,2)→q_norm→rope order, QK-norm on
  `[1,H,seq,D]`, SDPA layout, batch-1 drop, rope args. Sonnet
  sanity-check CONFIRMED CLEAN. **Strict stable 1/3.**
- **r11** — cc-qwen: CLEAN. Traced projection layout (GGUF ne
  reversal), layer_err closure lifetime, refutable-pattern
  question (single-variant enum, irrefutable), row_major_offset
  glue. Sonnet CONFIRMED CLEAN with one non-defect note about
  a comment "matches HF order" being numerically identical but
  minutely reordered (q_norm before vs after transpose — same
  result on last-axis RMSNorm). **Strict stable 2/3.**
- **r12** — cc-qwen: 1 finding (`Qwen3` is `Send + !Sync` because
  `Array` is `!Sync`). Sonnet WEAKENED: the module doc at
  `mod.rs:7` already says "no threading", so the constraint is
  already documented. No action. **Strict stable 3/3.**

## CONVERGED (strict)

Rounds 10, 11, 12 all had zero new items applied — three
consecutive strict-stable rounds. Per CLAUDE.md's convergence rule
under strict reading, the block is converged.

The r9 HIGH bug is the block's most important lesson: EIGHT rounds
of "clean" review missed a fundamental math error because the
fixture's uniform attn_v weights make V position-invariant, which
hides RoPE effects on the final logits. Only Sonnet's cold-read
sanity check (invited exactly because we no longer trusted the
"CLEAN" verdicts to see subtle things after many rounds) caught
it — by checking MLX C++ source directly for the axis convention.
The value-correctness gap warning at forward_tests.rs:94 was not
merely academic; it flagged a real class of bug that later
manifested. The loop chapter's HF-oracle golden test is now doubly
important: it is the only defense against this class of silent
wrong-logits bug we cannot unit-test.

## Final state (post-r12)

- 14 tests passing (12 family + 2 nucleon-mlx), including the
  new `rope_actually_rotates_along_seq_axis` revert-proof test
- Quality gate green
- All code fixes across 12 rounds revert-proof-verified
- Known coverage gap: value correctness of the forward pass
  (RoPE actually rotating with correct theta, QK-norm ordering,
  attn scale, GQA broadcast, tied-head matmul) is NOT asserted
  by this module's unit tests. The book explicitly acknowledges
  this and defers to the loop chapter's HF-oracle golden test.
  The r9 discovery **empirically confirms** this gap catches
  real bugs — eight of our review rounds missed one hiding
  behind the fixture math.

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
